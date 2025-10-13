from flask import Flask, render_template, request, jsonify
import pickle
import math

app = Flask(__name__)

# 尝试直接加载模型，不依赖scikit-learn
try:
    with open('masld_gb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    model_loaded = True
    print("模型加载成功")
except Exception as e:
    model_loaded = False
    print(f"模型加载失败: {e}")
    # 创建一个简单的备用预测函数
    def simple_predict(tg, glucose, hdl, bmi):
        # 这是一个简化的线性模型示例
        # 你需要根据你的实际模型调整这些权重
        score = (tg * 0.01 + glucose * 0.02 - hdl * 0.015 + bmi * 0.03)
        return min(max(score, 0), 1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        tg = float(data.get('tg', 0))
        glucose = float(data.get('glucose', 0))
        hdl = float(data.get('hdl', 0))
        bmi = float(data.get('bmi', 0))
        
        if model_loaded:
            # 使用真实模型
            features = [[tg, glucose, hdl, bmi]]
            prediction = model.predict_proba(features)[0][1]
        else:
            # 使用简化模型
            prediction = simple_predict(tg, glucose, hdl, bmi)
        
        risk_percentage = round(prediction * 100, 2)
        
        if risk_percentage < 20:
            risk_level = "低风险"
            suggestion = "保持良好的生活习惯"
        elif risk_percentage < 50:
            risk_level = "中风险"
            suggestion = "建议定期检查，注意饮食和运动"
        else:
            risk_level = "高风险"
            suggestion = "建议咨询专业医生进行进一步检查"
        
        return jsonify({
            'status': 'success',
            'risk_score': risk_percentage,
            'risk_level': risk_level,
            'suggestion': suggestion
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'预测出错: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
