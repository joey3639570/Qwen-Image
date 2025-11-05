"""
Multi-GPU Manager for Qwen-Image-Edit Pipeline
"""
import os
import time
import threading
import queue
import torch
import torch.multiprocessing as mp
from multiprocessing import Process, Queue, Event

mp.set_start_method('spawn', force=True)

from diffusers import QwenImageEditPipeline, QwenImageEditPlusPipeline


class EditGPUWorker:
    """GPU Worker for image editing tasks"""
    def __init__(self, gpu_id, model_repo_id, task_queue, result_queue, stop_event, use_plus_pipeline=False):
        self.gpu_id = gpu_id
        self.model_repo_id = model_repo_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = f"cuda:{gpu_id}"
        self.pipe = None
        self.use_plus_pipeline = use_plus_pipeline
        
    def initialize_model(self):
        """Initialize the QwenImageEditPipeline or QwenImageEditPlusPipeline on the specified GPU"""
        try:
            torch.cuda.set_device(self.gpu_id)
            if torch.cuda.is_available():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            
            if self.use_plus_pipeline:
                self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                    self.model_repo_id, 
                    torch_dtype=torch_dtype
                )
                print(f"GPU {self.gpu_id} initialized with QwenImageEditPlusPipeline (multi-image support)")
            else:
                self.pipe = QwenImageEditPipeline.from_pretrained(
                    self.model_repo_id, 
                    torch_dtype=torch_dtype
                )
                print(f"GPU {self.gpu_id} initialized with QwenImageEditPipeline (single image)")
            
            self.pipe = self.pipe.to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            print(f"GPU {self.gpu_id} model initialized successfully")
            return True
        except Exception as e:
            print(f"GPU {self.gpu_id} model initialization failed: {e}")
            return False
    
    def process_task(self, task):
        """Process an image editing task (supports both single and multiple images)"""
        try:
            task_id = task['task_id']
            images = task['images']  # Can be single image or list of images
            prompt = task['prompt']
            negative_prompt = task.get('negative_prompt', ' ')
            seed = task['seed']
            true_guidance_scale = task['true_guidance_scale']
            num_inference_steps = task['num_inference_steps']
            num_images_per_prompt = task.get('num_images_per_prompt', 1)
            guidance_scale = task.get('guidance_scale', 1.0)  # For plus pipeline
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            with torch.cuda.device(self.gpu_id):
                with torch.inference_mode():
                    if self.use_plus_pipeline and isinstance(images, list):
                        # Multi-image editing with QwenImageEditPlusPipeline
                        output = self.pipe(
                            image=images,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            true_cfg_scale=true_guidance_scale,
                            guidance_scale=guidance_scale,
                            num_images_per_prompt=num_images_per_prompt
                        )
                    else:
                        # Single image editing (convert list to single if needed)
                        single_image = images[0] if isinstance(images, list) else images
                        output = self.pipe(
                            image=single_image,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                            true_cfg_scale=true_guidance_scale,
                            num_images_per_prompt=num_images_per_prompt
                        )
                    
                    result_images = output.images if isinstance(output.images, list) else [output.images]
            
            return {
                'task_id': task_id,
                'images': result_images,
                'success': True,
                'gpu_id': self.gpu_id
            }
        except Exception as e:
            return {
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'gpu_id': self.gpu_id
            }
    
    def run(self):
        """Worker main loop"""
        if not self.initialize_model():
            return
        
        print(f"GPU {self.gpu_id} worker starting")
        
        while not self.stop_event.is_set():
            try:
                # Get task from the task queue, set timeout to check stop event
                task = self.task_queue.get(timeout=1)
                if task is None:  # Poison pill, exit signal
                    break
                
                # Process the task
                result = self.process_task(task)
                
                # Put the result into the result queue
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"GPU {self.gpu_id} worker exception: {e}")
                continue
        
        print(f"GPU {self.gpu_id} worker stopping")


# Global GPU worker function for spawn mode
def edit_gpu_worker_process(gpu_id, model_repo_id, task_queue, result_queue, stop_event, use_plus_pipeline=False):
    """Global function for multiprocessing spawn mode"""
    worker = EditGPUWorker(gpu_id, model_repo_id, task_queue, result_queue, stop_event, use_plus_pipeline)
    worker.run()


class EditMultiGPUManager:
    """Multi-GPU Manager for image editing tasks"""
    def __init__(self, model_repo_id="Qwen/Qwen-Image-Edit", num_gpus=None, task_queue_size=100, use_plus_pipeline=False):
        self.model_repo_id = model_repo_id
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.task_queue = Queue(maxsize=task_queue_size)
        self.result_queue = Queue()
        self.stop_event = Event()
        self.worker_processes = []
        self.task_counter = 0
        self.pending_tasks = {}
        self.use_plus_pipeline = use_plus_pipeline
        
        pipeline_type = "QwenImageEditPlusPipeline (multi-image)" if use_plus_pipeline else "QwenImageEditPipeline (single-image)"
        print(f"Initializing Multi-GPU Manager with {self.num_gpus} GPUs, queue size {task_queue_size}, pipeline: {pipeline_type}")
        
    def start_workers(self):
        """Start all GPU workers"""
        for gpu_id in range(self.num_gpus):
            process = Process(
                target=edit_gpu_worker_process,
                args=(gpu_id, self.model_repo_id, self.task_queue, 
                      self.result_queue, self.stop_event, self.use_plus_pipeline)
            )
            process.start()
            self.worker_processes.append(process)
        
        # Start result processing thread
        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()
        
        print(f"All {self.num_gpus} GPU workers have started")
    
    def _process_results(self):
        """Background thread for processing results"""
        while not self.stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=1)
                task_id = result['task_id']
                
                if task_id in self.pending_tasks:
                    # Pass the result to the waiting task
                    self.pending_tasks[task_id]['result'] = result
                    self.pending_tasks[task_id]['event'].set()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Result processing thread exception: {e}")
                continue
    
    def submit_task(self, images, prompt, negative_prompt=" ", seed=42, 
                   true_guidance_scale=4.0, num_inference_steps=50, 
                   num_images_per_prompt=1, guidance_scale=1.0, timeout=300):
        """Submit task and wait for result
        
        Args:
            images: Single image (PIL.Image) or list of images for multi-image editing
            prompt: Edit instruction prompt
            negative_prompt: Negative prompt (default: " ")
            seed: Random seed
            true_guidance_scale: True guidance scale
            num_inference_steps: Number of inference steps
            num_images_per_prompt: Number of output images per prompt
            guidance_scale: Guidance scale for plus pipeline (default: 1.0)
            timeout: Task timeout in seconds
        """
        task_id = f"task_{self.task_counter}_{time.time()}"
        self.task_counter += 1
        
        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]
        
        task = {
            'task_id': task_id,
            'images': images,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'seed': seed,
            'true_guidance_scale': true_guidance_scale,
            'num_inference_steps': num_inference_steps,
            'num_images_per_prompt': num_images_per_prompt,
            'guidance_scale': guidance_scale
        }
        
        # Create waiting event
        result_event = threading.Event()
        self.pending_tasks[task_id] = {
            'event': result_event,
            'result': None,
            'submitted_time': time.time()
        }
        
        try:
            # Put task into queue
            self.task_queue.put(task, timeout=10)
            
            # Wait for result
            start_time = time.time()
            while not result_event.is_set():
                if result_event.wait(timeout=2):  # Check every 2 seconds
                    break
                    
                if time.time() - start_time > timeout:
                    # Timeout
                    del self.pending_tasks[task_id]
                    return {'success': False, 'error': 'Task timeout'}
            
            result = self.pending_tasks[task_id]['result']
            del self.pending_tasks[task_id]
            return result
                
        except queue.Full:
            del self.pending_tasks[task_id]
            return {'success': False, 'error': 'Task queue is full'}
        except Exception as e:
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]
            return {'success': False, 'error': str(e)}
    
    def get_queue_status(self):
        """Get queue status"""
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(self.num_gpus):
                try:
                    gpu_info.append({
                        'gpu_id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_allocated': torch.cuda.memory_allocated(i) / 1024**3,  # GB
                        'memory_reserved': torch.cuda.memory_reserved(i) / 1024**3,  # GB
                    })
                except:
                    pass
        
        return {
            'task_queue_size': self.task_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'pending_tasks': len(self.pending_tasks),
            'active_workers': len(self.worker_processes),
            'total_gpus': self.num_gpus,
            'gpu_info': gpu_info
        }
    
    def stop(self):
        """Stop all workers"""
        print("Stopping Multi-GPU Manager...")
        self.stop_event.set()
        
        # Send stop signal to each worker
        for _ in range(self.num_gpus):
            try:
                self.task_queue.put(None, timeout=1)
            except queue.Full:
                pass
        
        # Wait for all processes to end
        for process in self.worker_processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        
        print("Multi-GPU Manager has stopped")

