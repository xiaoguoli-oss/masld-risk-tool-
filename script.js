// 显示结果函数
function showResult(message, type) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = message;
    resultDiv.className = 'result ' + type;
    resultDiv.style.display = 'block';
    
    // 滚动到结果区域
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// 验证输入数据
function validateInputs(tg, glucose, hdl, bmi) {
    const errors = [];
    
    if (!tg || tg <= 0) errors.push('请输入有效的甘油三酯值');
    if (!glucose || glucose <= 0) errors.push('请输入有效的血糖值');
    if (!hdl || hdl <= 0) errors.push('请输入有效的HDL值');
    if (!bmi || bmi <= 0) errors.push('请输入有效的BMI值');
    
    if (tg > 1000) errors.push('甘油三酯值过高，请检查输入');
    if (glucose > 500) errors.push('血糖值过高，请检查输入');
    if (hdl > 200) errors.push('HDL值过高，请检查输入');
    if (bmi > 50) errors.push('BMI值过高，请检查输入');
    
    return errors;
}

// 主要计算函数
async function calculateRisk() {
    try {
        // 获取输入值
        const tg = document.getElementById('tg').value;
        const glucose = document.getElementById('glucose').value;
        const hdl = document.getElementById('hdl').value;
        const bmi = document.getElementById('bmi').value;
        
        // 验证输入
        const validationErrors = validateInputs(tg, glucose, hdl, bmi);
        if (validationErrors.length > 0) {
            showResult(`
                <h3>输入错误</h3>
                <ul>
                    ${validationErrors.map(error => `<li>${error}</li>`).join('')}
                </ul>
            `, 'error');
            return;
        }
        
        // 显示加载中
        showResult('正在计算风险评估，请稍候...', 'loading');
        
        // 发送数据到后端API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                tg: parseFloat(tg),
                glucose: parseFloat(glucose),
                hdl: parseFloat(hdl),
                bmi: parseFloat(bmi)
            })
        });
        
        if (!response.ok) {
            throw new Error(`网络请求失败: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showResult(`
                <h3>🔍 风险评估结果</h3>
                <p><strong>风险分数:</strong> ${data.risk_score}%</p>
                <p><strong>风险等级:</strong> ${data.risk_level}</p>
                <p><strong>健康建议:</strong> ${data.suggestion}</p>
                ${data.risk_score > 50 ? '<p style="color: #dc3545; margin-top: 15px;">⚠️ 建议尽快咨询专业医生</p>' : ''}
            `, 'success');
        } else {
            showResult(`
                <h3>❌ 计算错误</h3>
                <p>${data.message}</p>
                <p style="margin-top: 10px; font-size: 14px;">请检查输入数据或稍后重试</p>
            `, 'error');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showResult(`
            <h3>❌ 网络错误</h3>
            <p>${error.message}</p>
            <p style="margin-top: 10px; font-size: 14px;">请检查网络连接后重试</p>
        `, 'error');
    }
}

// 输入框回车键支持
document.addEventListener('DOMContentLoaded', function() {
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
        input.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                calculateRisk();
            }
        });
    });
    
    // 页面加载完成后的提示
    console.log('MASLD风险评估工具已加载完成');
});
