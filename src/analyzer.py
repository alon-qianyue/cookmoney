"""
AI Analyzer module using OpenAI API
"""
import json
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL, ANALYSIS_DIR
from .file_processor import FileProcessor


class AIAnalyzer:
    """Analyze research reports using OpenAI API"""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.file_processor = FileProcessor()
    
    def _prepare_messages_for_text(self, processed_files: List[Dict]) -> List[Dict]:
        """Prepare messages for text-based analysis"""
        messages = [
            {
                "role": "system",
                "content": """你是一位资深的投资分析师，专门分析各类投资研报、市场分析报告等文件。
你的任务是：
1. 仔细阅读并理解所有提供的研报内容
2. 提取关键信息：投资标的、行业趋势、风险因素、机会点等
3. 综合分析多份报告的观点，找出共识和分歧
4. 基于分析给出清晰、可执行的投资建议
5. 评估投资风险等级（低、中、高）

请用中文回复，回答要专业、客观、有深度。"""
            }
        ]
        
        # Prepare content from all files
        content_parts = []
        for idx, file_data in enumerate(processed_files):
            if "error" in file_data:
                content_parts.append(f"文件 {idx+1} ({file_data['file_name']}): 处理错误 - {file_data['error']}")
                continue
            
            if file_data["type"] == "text":
                content_parts.append(f"""
=== 研报文件 {idx+1}: {file_data['file_name']} ===
格式: {file_data['format']}
内容:
{file_data['content'][:10000]}  # Limit to first 10000 chars per file
""")
        
        messages.append({
            "role": "user",
            "content": "\n\n".join(content_parts) + "\n\n请基于以上研报内容，提供综合投资分析和建议。"
        })
        
        return messages
    
    def _prepare_messages_for_images(self, processed_files: List[Dict]) -> List[Dict]:
        """Prepare messages for image-based analysis"""
        messages = [
            {
                "role": "system",
                "content": """你是一位资深的投资分析师，专门分析各类投资研报图表、市场数据可视化等图片。
你的任务是：
1. 仔细分析图片中的数据、图表、趋势
2. 提取关键信息：价格走势、成交量、技术指标等
3. 识别重要的模式和信号
4. 基于图表分析给出投资建议
5. 评估投资风险等级（低、中、高）

请用中文回复，回答要专业、客观、有深度。"""
            }
        ]
        
        # Prepare content with images
        content_parts = [{"type": "text", "text": "请分析以下投资研报图片，并提供综合投资建议：\n"}]
        
        for idx, file_data in enumerate(processed_files):
            if "error" in file_data:
                content_parts.append({
                    "type": "text",
                    "text": f"\n文件 {idx+1} ({file_data['file_name']}): 处理错误 - {file_data['error']}\n"
                })
                continue
            
            if file_data["type"] == "image":
                content_parts.append({
                    "type": "text",
                    "text": f"\n图片 {idx+1}: {file_data['file_name']} ({file_data['format']}, {file_data['width']}x{file_data['height']})\n"
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{file_data['mime_type']};base64,{file_data['data']}"
                    }
                })
        
        messages.append({
            "role": "user",
            "content": content_parts
        })
        
        return messages
    
    def _prepare_messages_mixed(self, processed_files: List[Dict]) -> List[Dict]:
        """Prepare messages for mixed content (text + images)"""
        messages = [
            {
                "role": "system",
                "content": """你是一位资深的投资分析师，专门分析各类投资研报、市场分析报告等文件（包括文本和图表）。
你的任务是：
1. 仔细阅读文本内容并分析图表数据
2. 提取关键信息：投资标的、行业趋势、风险因素、技术指标等
3. 综合文字分析和图表数据，找出关键观点
4. 基于综合分析给出清晰、可执行的投资建议
5. 评估投资风险等级（低、中、高）

请用中文回复，回答要专业、客观、有深度。"""
            }
        ]
        
        # Prepare mixed content
        content_parts = [{"type": "text", "text": "请分析以下投资研报内容（包括文本和图片），并提供综合投资建议：\n\n"}]
        
        for idx, file_data in enumerate(processed_files):
            if "error" in file_data:
                content_parts.append({
                    "type": "text",
                    "text": f"\n文件 {idx+1} ({file_data['file_name']}): 处理错误 - {file_data['error']}\n"
                })
                continue
            
            if file_data["type"] == "image":
                content_parts.append({
                    "type": "text",
                    "text": f"\n=== 图片 {idx+1}: {file_data['file_name']} ===\n"
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{file_data['mime_type']};base64,{file_data['data']}"
                    }
                })
            elif file_data["type"] == "text":
                content_parts.append({
                    "type": "text",
                    "text": f"\n=== 文本 {idx+1}: {file_data['file_name']} ===\n{file_data['content'][:10000]}\n"
                })
        
        messages.append({
            "role": "user",
            "content": content_parts
        })
        
        return messages
    
    def analyze_reports(self, file_paths: List[Path]) -> Dict:
        """Analyze multiple research reports and generate investment recommendations"""
        # Process all files
        processed_files = self.file_processor.process_multiple_files(file_paths)
        
        # Determine content type
        has_images = any(f.get("type") == "image" for f in processed_files if "error" not in f)
        has_text = any(f.get("type") == "text" for f in processed_files if "error" not in f)
        
        # Prepare messages based on content type
        if has_images and has_text:
            messages = self._prepare_messages_mixed(processed_files)
        elif has_images:
            messages = self._prepare_messages_for_images(processed_files)
        else:
            messages = self._prepare_messages_for_text(processed_files)
        
        # Call OpenAI API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "files_analyzed": [f.get("file_name") for f in processed_files],
                "model": self.model,
                "analysis": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "processed_files": processed_files
            }
            
            return analysis_result
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "files_analyzed": [f.get("file_name") for f in processed_files],
                "error": str(e),
                "processed_files": processed_files
            }
    
    def save_analysis(self, analysis_result: Dict, output_name: Optional[str] = None) -> Path:
        """Save analysis result to a file"""
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"analysis_{timestamp}.json"
        
        output_path = ANALYSIS_DIR / output_name
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        # Also save a markdown version for easier reading
        md_path = output_path.with_suffix(".md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 投资研报分析报告\n\n")
            f.write(f"**分析时间**: {analysis_result['timestamp']}\n\n")
            f.write(f"**分析文件**: {', '.join(analysis_result['files_analyzed'])}\n\n")
            
            if "error" in analysis_result:
                f.write(f"## 错误\n\n{analysis_result['error']}\n")
            else:
                f.write(f"**使用模型**: {analysis_result['model']}\n\n")
                if analysis_result.get('tokens_used'):
                    f.write(f"**Token使用量**: {analysis_result['tokens_used']}\n\n")
                f.write(f"## 分析结果\n\n{analysis_result['analysis']}\n")
        
        return output_path
