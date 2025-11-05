"""
Qwen Image Edit Service - 16種編輯模式的提示詞生成和處理
"""
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
import enum


class EditMode(enum.Enum):
    """16種編輯模式枚舉"""
    # 多圖編輯模式 (1-5)
    PERSON_PERSON = 1  # 人物 + 人物
    PERSON_PRODUCT = 2  # 人物 + 產品
    PERSON_SCENE = 3  # 人物 + 場景
    PRODUCT_PRODUCT = 4  # 產品 + 產品
    MULTI_OBJECT = 5  # 多物件組合
    
    # 單圖一致性模式 (6-8)
    PERSON_CONSISTENCY = 6  # 人物一致性編輯
    PRODUCT_CONSISTENCY = 7  # 產品一致性編輯
    STYLE_TRANSFER = 8  # 風格轉換
    
    # 文字編輯模式 (9-12)
    TEXT_REPLACE = 9  # 文字替換
    TEXT_ADD = 10  # 文字添加
    TEXT_FONT = 11  # 文字字體編輯
    TEXT_COLOR = 12  # 文字顏色編輯
    
    # ControlNet 模式 (13-16)
    DEPTH_CONTROL = 13  # 深度圖控制
    EDGE_CONTROL = 14  # 邊緣圖控制
    KEYPOINT_CONTROL = 15  # 關鍵點控制
    SKETCH_CONTROL = 16  # 草圖控制


class PromptGenerator:
    """提示詞生成器 - 根據編輯模式生成對應的提示詞"""
    
    @staticmethod
    def generate_prompt(
        mode: EditMode,
        user_prompt: str,
        images_count: int,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """根據編輯模式生成提示詞"""
        additional_params = additional_params or {}
        
        if mode == EditMode.PERSON_PERSON:
            return PromptGenerator._generate_person_person_prompt(user_prompt, images_count, additional_params)
        elif mode == EditMode.PERSON_PRODUCT:
            return PromptGenerator._generate_person_product_prompt(user_prompt, images_count, additional_params)
        elif mode == EditMode.PERSON_SCENE:
            return PromptGenerator._generate_person_scene_prompt(user_prompt, images_count, additional_params)
        elif mode == EditMode.PRODUCT_PRODUCT:
            return PromptGenerator._generate_product_product_prompt(user_prompt, images_count, additional_params)
        elif mode == EditMode.MULTI_OBJECT:
            return PromptGenerator._generate_multi_object_prompt(user_prompt, images_count, additional_params)
        elif mode == EditMode.PERSON_CONSISTENCY:
            return PromptGenerator._generate_person_consistency_prompt(user_prompt, additional_params)
        elif mode == EditMode.PRODUCT_CONSISTENCY:
            return PromptGenerator._generate_product_consistency_prompt(user_prompt, additional_params)
        elif mode == EditMode.STYLE_TRANSFER:
            return PromptGenerator._generate_style_transfer_prompt(user_prompt, additional_params)
        elif mode == EditMode.TEXT_REPLACE:
            return PromptGenerator._generate_text_replace_prompt(user_prompt, additional_params)
        elif mode == EditMode.TEXT_ADD:
            return PromptGenerator._generate_text_add_prompt(user_prompt, additional_params)
        elif mode == EditMode.TEXT_FONT:
            return PromptGenerator._generate_text_font_prompt(user_prompt, additional_params)
        elif mode == EditMode.TEXT_COLOR:
            return PromptGenerator._generate_text_color_prompt(user_prompt, additional_params)
        elif mode == EditMode.DEPTH_CONTROL:
            return PromptGenerator._generate_depth_control_prompt(user_prompt, additional_params)
        elif mode == EditMode.EDGE_CONTROL:
            return PromptGenerator._generate_edge_control_prompt(user_prompt, additional_params)
        elif mode == EditMode.KEYPOINT_CONTROL:
            return PromptGenerator._generate_keypoint_control_prompt(user_prompt, additional_params)
        elif mode == EditMode.SKETCH_CONTROL:
            return PromptGenerator._generate_sketch_control_prompt(user_prompt, additional_params)
        else:
            return user_prompt
    
    @staticmethod
    def _generate_person_person_prompt(prompt: str, images_count: int, params: Dict) -> str:
        """人物 + 人物模式提示詞"""
        positions = params.get('positions', ['left', 'right'])
        if images_count == 2:
            return f"{prompt}. The person from the first image is on the {positions[0]}, the person from the second image is on the {positions[1]}, facing each other."
        return f"{prompt}. Multiple people from the input images are arranged in the scene."
    
    @staticmethod
    def _generate_person_product_prompt(prompt: str, images_count: int, params: Dict) -> str:
        """人物 + 產品模式提示詞"""
        return f"{prompt}. The person is interacting with the product, maintaining the identity of both."
    
    @staticmethod
    def _generate_person_scene_prompt(prompt: str, images_count: int, params: Dict) -> str:
        """人物 + 場景模式提示詞"""
        return f"{prompt}. The person is placed in the scene, maintaining their identity and appearance."
    
    @staticmethod
    def _generate_product_product_prompt(prompt: str, images_count: int, params: Dict) -> str:
        """產品 + 產品模式提示詞"""
        return f"{prompt}. Multiple products are arranged together, maintaining their individual identities."
    
    @staticmethod
    def _generate_multi_object_prompt(prompt: str, images_count: int, params: Dict) -> str:
        """多物件組合模式提示詞"""
        return f"{prompt}. Multiple objects from the input images are combined in a cohesive scene."
    
    @staticmethod
    def _generate_person_consistency_prompt(prompt: str, params: Dict) -> str:
        """人物一致性模式提示詞"""
        preserve_features = params.get('preserve_features', ['face', 'hair', 'body'])
        return f"{prompt}. Maintain the person's identity: {', '.join(preserve_features)}."
    
    @staticmethod
    def _generate_product_consistency_prompt(prompt: str, params: Dict) -> str:
        """產品一致性模式提示詞"""
        return f"{prompt}. Maintain the product's identity, shape, and key features."
    
    @staticmethod
    def _generate_style_transfer_prompt(prompt: str, params: Dict) -> str:
        """風格轉換模式提示詞"""
        style = params.get('style', 'realistic')
        return f"{prompt}. Apply {style} style while maintaining the subject's identity."
    
    @staticmethod
    def _generate_text_replace_prompt(prompt: str, params: Dict) -> str:
        """文字替換模式提示詞"""
        old_text = params.get('old_text', '')
        new_text = params.get('new_text', '')
        if old_text and new_text:
            return f'Replace "{old_text}" to "{new_text}". {prompt}'
        return f"Replace text content. {prompt}"
    
    @staticmethod
    def _generate_text_add_prompt(prompt: str, params: Dict) -> str:
        """文字添加模式提示詞"""
        text = params.get('text', '')
        position = params.get('position', 'center')
        if text:
            return f'Add text "{text}" at the {position}. {prompt}'
        return f"Add text. {prompt}"
    
    @staticmethod
    def _generate_text_font_prompt(prompt: str, params: Dict) -> str:
        """文字字體編輯模式提示詞"""
        font_type = params.get('font_type', 'default')
        return f"Change text font to {font_type}. {prompt}"
    
    @staticmethod
    def _generate_text_color_prompt(prompt: str, params: Dict) -> str:
        """文字顏色編輯模式提示詞"""
        color = params.get('color', 'black')
        return f"Change text color to {color}. {prompt}"
    
    @staticmethod
    def _generate_depth_control_prompt(prompt: str, params: Dict) -> str:
        """深度圖控制模式提示詞"""
        return f"{prompt}. Follow the depth map structure."
    
    @staticmethod
    def _generate_edge_control_prompt(prompt: str, params: Dict) -> str:
        """邊緣圖控制模式提示詞"""
        return f"{prompt}. Follow the edge map structure."
    
    @staticmethod
    def _generate_keypoint_control_prompt(prompt: str, params: Dict) -> str:
        """關鍵點控制模式提示詞"""
        return f"{prompt}. Follow the keypoint pose structure."
    
    @staticmethod
    def _generate_sketch_control_prompt(prompt: str, params: Dict) -> str:
        """草圖控制模式提示詞"""
        return f"{prompt}. Follow the sketch structure."


class MultiImageEditService:
    """多圖片編輯服務"""
    
    def __init__(self):
        self.prompt_generator = PromptGenerator()
    
    def validate_parameters(
        self,
        mode: int,
        images: List[Image.Image],
        controlnet_image: Optional[Image.Image] = None,
        text_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """驗證參數"""
        # 驗證模式
        if mode < 1 or mode > 16:
            return False, "編輯模式必須在 1-16 之間"
        
        # 驗證圖片數量
        if len(images) < 1:
            return False, "至少需要 1 張圖片"
        if len(images) > 5:
            return False, "最多支援 5 張圖片"
        
        # 驗證多圖模式需要至少 2 張圖片
        if mode <= 5 and len(images) < 2:
            return False, f"模式 {mode} 需要至少 2 張圖片"
        
        # 驗證 ControlNet 模式需要條件圖
        if mode >= 13 and mode <= 16:
            if controlnet_image is None:
                return False, f"模式 {mode} 需要 ControlNet 條件圖"
        
        # 驗證文字編輯模式需要文字參數
        if mode >= 9 and mode <= 12:
            if text_params is None:
                return False, f"模式 {mode} 需要文字編輯參數"
        
        return True, None
    
    def get_default_parameters(self, mode: int) -> Dict[str, Any]:
        """獲取模式的預設參數"""
        defaults = {
            'true_guidance_scale': 4.0,
            'guidance_scale': 1.0,
            'num_inference_steps': 40,
            'num_images_per_prompt': 1,
            'seed': None,
        }
        
        # 根據模式調整預設值
        if mode >= 13 and mode <= 16:  # ControlNet 模式
            defaults['num_inference_steps'] = 50
        elif mode >= 9 and mode <= 12:  # 文字編輯模式
            defaults['num_inference_steps'] = 50
        
        return defaults
    
    def process_multi_image_edit(
        self,
        mode: int,
        images: List[Image.Image],
        user_prompt: str,
        controlnet_image: Optional[Image.Image] = None,
        text_params: Optional[Dict[str, Any]] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        處理多圖片編輯請求
        
        注意：此函數目前返回模擬回應，待模型端點完成後需要實作實際 API 呼叫
        """
        # 驗證參數
        is_valid, error_msg = self.validate_parameters(mode, images, controlnet_image, text_params)
        if not is_valid:
            return {
                'success': False,
                'error': error_msg
            }
        
        # 獲取編輯模式
        try:
            edit_mode = EditMode(mode)
        except ValueError:
            return {
                'success': False,
                'error': f'無效的編輯模式: {mode}'
            }
        
        # 生成提示詞
        prompt_params = {}
        if text_params:
            prompt_params.update(text_params)
        if additional_params:
            prompt_params.update(additional_params)
        
        enhanced_prompt = self.prompt_generator.generate_prompt(
            edit_mode,
            user_prompt,
            len(images),
            prompt_params
        )
        
        # 獲取預設參數
        default_params = self.get_default_parameters(mode)
        
        # TODO: 實作實際的模型 API 呼叫
        # 目前返回模擬回應
        return {
            'success': True,
            'prompt': enhanced_prompt,
            'mode': mode,
            'images_count': len(images),
            'has_controlnet': controlnet_image is not None,
            'has_text_params': text_params is not None,
            'default_params': default_params,
            'message': '模擬回應：待模型端點完成後將實作實際處理'
        }
    
    def get_mode_info(self, mode: int) -> Optional[Dict[str, Any]]:
        """獲取編輯模式資訊"""
        mode_names = {
            1: "人物 + 人物",
            2: "人物 + 產品",
            3: "人物 + 場景",
            4: "產品 + 產品",
            5: "多物件組合",
            6: "人物一致性編輯",
            7: "產品一致性編輯",
            8: "風格轉換",
            9: "文字替換",
            10: "文字添加",
            11: "文字字體編輯",
            12: "文字顏色編輯",
            13: "深度圖控制",
            14: "邊緣圖控制",
            15: "關鍵點控制",
            16: "草圖控制",
        }
        
        mode_descriptions = {
            1: "將兩個人物合併到同一個場景中",
            2: "將人物和產品組合在一起",
            3: "將人物放置在場景中",
            4: "將多個產品組合在一起",
            5: "將多個物件組合在一起",
            6: "編輯人物時保持身份一致性",
            7: "編輯產品時保持產品一致性",
            8: "將圖片轉換為不同風格",
            9: "替換圖片中的文字",
            10: "在圖片中添加文字",
            11: "更改文字的字體",
            12: "更改文字的顏色",
            13: "使用深度圖控制生成結果",
            14: "使用邊緣圖控制生成結果",
            15: "使用關鍵點控制生成結果",
            16: "使用草圖控制生成結果",
        }
        
        if mode < 1 or mode > 16:
            return None
        
        return {
            'mode': mode,
            'name': mode_names.get(mode, '未知模式'),
            'description': mode_descriptions.get(mode, ''),
            'requires_multiple_images': mode <= 5,
            'requires_controlnet': mode >= 13 and mode <= 16,
            'requires_text_params': mode >= 9 and mode <= 12,
            'min_images': 2 if mode <= 5 else 1,
            'max_images': 5
        }

