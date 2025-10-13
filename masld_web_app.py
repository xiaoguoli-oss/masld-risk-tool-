import math
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
import os
import json

app = Flask(__name__)

# 全局变量存储模型信息
model_info = None
feature_columns = []

def load_model():
    """加载训练好的模型"""
    global model_info, feature_columns
    
    try:
        if not os.path.exists('masld_gb_model.pkl'):
            return False, "Model file not found, please train and save the model first"
        
        model_info = joblib.load('masld_gb_model.pkl')
        feature_columns = model_info['feature_columns']
        
        print(f"Model loaded successfully, using {len(feature_columns)} features")
        return True, "Model loaded successfully"
        
    except Exception as e:
        return False, f"Model loading failed: {str(e)}"

def calculate_medical_indexes(input_data):
    """计算医学指标 - 修复版本"""
    # 提取基础指标
    tg = float(input_data['tg'])  # mg/dl
    glucose = float(input_data['glucose'])  # mg/dl
    hdl = float(input_data['hdl'])  # mg/dl
    bmi = float(input_data['bmi'])  # kg/m²
    
    # 验证输入值
    if tg <= 0 or glucose <= 0 or hdl <= 0 or bmi <= 0:
        raise ValueError("All input values must be positive numbers")
    
    try:
        # 计算TyG指数 
        tyg_index = math.log((tg * glucose) / 2)
        
        # 计算SPISE指数 - 修复版本
        # SPISE = (600 × HDL^0.185) / (TG^0.2 × BMI^1.338)
        # 确保所有计算都是正数
        hdl_power = max(hdl ** 0.185, 1e-10)  # 避免过小的值
        tg_power = max(tg ** 0.2, 1e-10)      # 避免过小的值
        bmi_power = max(bmi ** 1.338, 1e-10)  # 避免过小的值
        
        numerator = 600 * hdl_power
        denominator = tg_power * bmi_power
        
        if denominator <= 0:
            raise ValueError("Denominator in SPISE calculation must be positive")
            
        spise = numerator / denominator
        
        # 确保SPISE是正数
        if spise <= 0:
            raise ValueError("SPISE must be positive")
        
        # 计算METS-IR指数
        # METS-IR = Ln[(2 × glucose) + TG] × BMI / Ln(HDL)
        mets_ir_numerator = math.log((2 * glucose) + tg) * bmi
        
        # 确保HDL > 1，因为ln(HDL)在HDL<=1时为非正数
        if hdl <= 1:
            raise ValueError("HDL must be greater than 1 for METS-IR calculation")
            
        mets_ir_denominator = math.log(hdl)
        
        if mets_ir_denominator == 0:
            raise ValueError("Denominator in METS-IR calculation is zero")
            
        mets_ir = mets_ir_numerator / mets_ir_denominator
        
        # 计算TG/HDL比值
        tg_hdl_ratio = tg / hdl
        
        return {
            'SPISE': spise,
            'METS-IR': mets_ir,
            'TyG': tyg_index,
            'TG/HDL': tg_hdl_ratio,
            'input_values': {
                'TG': tg,
                'Glucose': glucose,
                'HDL': hdl,
                'BMI': bmi
            }
        }
        
    except ValueError as e:
        raise ValueError(f"Mathematical error in calculation: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error in calculation: {str(e)}")

def create_features_from_basic(input_data):
    """从基础指标创建完整特征集"""
    # 首先计算医学指标
    medical_indexes = calculate_medical_indexes(input_data)
    
    # 创建DataFrame
    df = pd.DataFrame([medical_indexes])
    
    # 基础特征
    base_features = ['SPISE', 'METS-IR', 'TyG', 'TG/HDL']
    
    # 创建交互特征
    df['SPISE_METS_interaction'] = df['SPISE'] * df['METS-IR']
    df['SPISE_METS_ratio'] = np.where(
        df['METS-IR'] != 0, 
        df['SPISE'] / df['METS-IR'], 
        0
    )
    df['SPISE_TyG_interaction'] = df['SPISE'] * df['TyG']
    df['METS_TyG_interaction'] = df['METS-IR'] * df['TyG']
    
    # 创建多项式特征
    for feature in base_features:
        df[f'{feature}_squared'] = df[feature] ** 2
    
    # 创建加权组合 - 使用固定的均值和标准差避免NaN
    # 使用训练数据的典型值作为参考
    spise_ref_mean, spise_ref_std = 6.5, 1.2   # 典型的SPISE均值和标准差
    mets_ir_ref_mean, mets_ir_ref_std = 35.0, 5.0  # 典型的METS-IR均值和标准差
    
    spise_norm = (df['SPISE'] - spise_ref_mean) / spise_ref_std
    mets_ir_norm = (df['METS-IR'] - mets_ir_ref_mean) / mets_ir_ref_std
    
    spise_weight, mets_ir_weight = 0.6, 0.4
    df['weighted_combination'] = spise_weight * spise_norm + mets_ir_weight * mets_ir_norm
    
    return df, medical_indexes

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """预测MASLD风险"""
    try:
        if model_info is None:
            return jsonify({
                'success': False, 
                'error': 'Model not loaded, please check model files'
            })
        
        # 获取用户输入
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data received'})
        
        # 验证输入
        required_fields = ['tg', 'glucose', 'hdl', 'bmi']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({
                    'success': False, 
                    'error': f'Missing required field: {field}'
                })
        
        # 转换为float并验证
        try:
            tg = float(data['tg'])
            glucose = float(data['glucose'])
            hdl = float(data['hdl'])
            bmi = float(data['bmi'])
        except ValueError:
            return jsonify({
                'success': False, 
                'error': 'All inputs must be valid numbers'
            })
        
        # 验证输入范围
        if tg <= 0 or glucose <= 0 or hdl <= 0 or bmi <= 0:
            return jsonify({
                'success': False, 
                'error': 'All values must be positive numbers'
            })
        
        # 特别验证HDL > 1（METS-IR计算需要）
        if hdl <= 1:
            return jsonify({
                'success': False, 
                'error': 'HDL must be greater than 1 for accurate calculation'
            })
        
        # 创建特征
        features_df, medical_indexes = create_features_from_basic(data)
        
        # 确保特征顺序与训练时一致
        X_pred = features_df[feature_columns].values
        
        # 预测
        model = model_info['model']
        probability = model.predict_proba(X_pred)[0, 1]
        risk_percentage = probability * 100
        
        # 风险等级判断
        if risk_percentage < 10:
            risk_level = "Very Low Risk"
            recommendation = "Maintain healthy lifestyle, regular check-ups"
            color_class = "risk-very-low"
        elif risk_percentage < 30:
            risk_level = "Low Risk" 
            recommendation = "Maintain good living habits, balanced diet"
            color_class = "risk-low"
        elif risk_percentage < 50:
            risk_level = "Medium Risk"
            recommendation = "Suggest improving lifestyle, increasing exercise, controlling weight"
            color_class = "risk-medium"
        elif risk_percentage < 70:
            risk_level = "Medium-High Risk"
            recommendation = "Recommend medical consultation and liver-related examinations"
            color_class = "risk-medium-high"
        else:
            risk_level = "High Risk"
            recommendation = "Strongly recommend immediate medical evaluation and intervention"
            color_class = "risk-high"
        
        return jsonify({
            'success': True,
            'risk_percentage': round(risk_percentage, 1),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'probability': round(probability, 3),
            'color_class': color_class,
            'calculated_indexes': {
                'SPISE': round(medical_indexes['SPISE'], 2),
                'METS_IR': round(medical_indexes['METS-IR'], 2),
                'TyG': round(medical_indexes['TyG'], 2),
                'TG_HDL': round(medical_indexes['TG/HDL'], 2)
            },
            'input_values': medical_indexes['input_values']
        })
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # 打印错误到控制台
        return jsonify({'success': False, 'error': str(e)})

@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html')

if __name__ == '__main__':
    success, message = load_model()
    print(message)
    
    if success:
        print("Starting MASLD Risk Assessment Web Application...")
        print("Access address: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to start application:", message)