from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np

app = Flask(__name__)

# 加载模型
print("正在加载MASLD预测模型...")
try:
    with open('masld_gb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('model_features.json', 'r') as f:
        feature_config = json.load(f)
    
    model_loaded = True
    print("✅ 模型加载成功!")
except Exception as e:
    model_loaded = False
    print(f"❌ 模型加载失败: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'status': 'error', 'message': '模型未正确加载'})
    
    try:
        data = request.json
        tg = float(data.get('tg', 0))
        glucose = float(data.get('glucose', 0))
        hdl = float(data.get('hdl', 0))
        bmi = float(data.get('bmi', 0))
        
        # 准备特征并预测
        features = np.array([[tg, glucose, hdl, bmi]])
        prediction = model.predict_proba(features)[0][1]
        risk_percentage = round(prediction * 100, 2)
        
        # 风险评估
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

# Vercel需要这个
if __name__ == '__main__':
    app.run(debug=True)
