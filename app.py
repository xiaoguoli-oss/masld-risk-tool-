from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# 加载模型
try:
    with open('masld_gb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    model_loaded = True
    print("✅ MASLD预测模型加载成功!")
except Exception as e:
    model_loaded = False
    print(f"❌ 模型加载失败: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'status': 'error', 'message': '模型未正确加载，请联系管理员'})
    
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
            suggestion = "保持良好的生活习惯，定期体检"
        elif risk_percentage < 50:
            risk_level = "中风险"
            suggestion = "建议改善饮食和增加运动，定期复查"
        else:
            risk_level = "高风险"
            suggestion = "强烈建议咨询专业医生进行详细检查"
        
        return jsonify({
            'status': 'success',
            'risk_score': risk_percentage,
            'risk_level': risk_level,
            'suggestion': suggestion
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'计算出错: {str(e)}'})

# PythonAnywhere需要这个
if __name__ == '__main__':
    app.run(debug=True)
