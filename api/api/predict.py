from http.server import BaseHTTPRequestHandler
import json
import pickle
import numpy as np
import os
import sys

# 添加当前目录到Python路径，确保可以导入模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        # 处理预检请求
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        try:
            # 设置CORS头
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            # 读取请求数据
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            if not post_data:
                response = {
                    'status': 'error',
                    'message': '未接收到数据'
                }
                self.wfile.write(json.dumps(response).encode())
                return
                
            data = json.loads(post_data)
            
            # 验证必需字段
            required_fields = ['tg', 'glucose', 'hdl', 'bmi']
            for field in required_fields:
                if field not in data:
                    response = {
                        'status': 'error',
                        'message': f'缺少必需字段: {field}'
                    }
                    self.wfile.write(json.dumps(response).encode())
                    return
            
            # 提取特征
            tg = float(data['tg'])
            glucose = float(data['glucose'])
            hdl = float(data['hdl'])
            bmi = float(data['bmi'])
            
            # 加载模型并进行预测
            try:
                # 注意：模型文件应该与predict.py在同一目录或在可访问的路径
                model_path = os.path.join(os.path.dirname(__file__), 'masld_gb_model.pkl')
                
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # 准备特征数据
                features = np.array([[tg, glucose, hdl, bmi]])
                
                # 进行预测
                prediction = model.predict_proba(features)[0][1]
                risk_percentage = round(prediction * 100, 2)
                
                # 风险评估逻辑
                if risk_percentage < 20:
                    risk_level = "低风险"
                    suggestion = "保持良好的生活习惯，均衡饮食和适量运动"
                elif risk_percentage < 50:
                    risk_level = "中风险" 
                    suggestion = "建议定期检查，注意饮食控制和增加体育锻炼"
                else:
                    risk_level = "高风险"
                    suggestion = "强烈建议咨询专业医生进行详细检查和评估"
                
                # 返回成功响应
                response = {
                    'status': 'success',
                    'risk_score': risk_percentage,
                    'risk_level': risk_level,
                    'suggestion': suggestion
                }
                
            except FileNotFoundError:
                response = {
                    'status': 'error',
                    'message': '机器学习模型文件未找到'
                }
            except Exception as model_error:
                response = {
                    'status': 'error',
                    'message': f'模型预测错误: {str(model_error)}'
                }
            
            # 发送响应
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode())
            
        except json.JSONDecodeError:
            response = {
                'status': 'error',
                'message': 'JSON数据格式错误'
            }
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            response = {
                'status': 'error',
                'message': f'服务器内部错误: {str(e)}'
            }
            self.wfile.write(json.dumps(response).encode())

# Vercel需要这个函数
def main(request, response):
    # 这个函数是为了兼容Vercel的服务器less函数格式
    # 实际处理仍然在Handler类中完成
    pass
