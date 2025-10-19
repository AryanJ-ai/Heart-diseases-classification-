import numpy as np
import pickle
import streamlit as st
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1ABC9C;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #F0F8FF 0%, #FFFFFF 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(26, 188, 156, 0.2);
        border: 1px solid rgba(26, 188, 156, 0.3);
    }
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
    }
    .risk-low {
        background-color: #E8F5E9;
        border-left: 5px solid #1ABC9C;
    }
    .risk-high {
        background-color: #FFEBEE;
        border-left: 5px solid #E74C3C;
    }
    .stApp {
        background-color: #F5F5F5;
    }
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #2C3E50;
    }
    .stButton>button {
        background-color: #1ABC9C;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #16A085;
        box-shadow: 0 4px 8px rgba(26, 188, 156, 0.3);
    }
    /* Style input fields for light theme */
    .stNumberInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #FFFFFF;
        color: #2C3E50;
        border: 1px solid #E0E0E0;
    }
    /* Style cards and containers */
    [data-testid="stHorizontalBlock"], [data-testid="column"] {
        background-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with error handling
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = pickle.load(open('C:\\Users\\aryan\\OneDrive\\Desktop\\project-2\\heart_disease_model.sav', 'rb'))
      
        return model
    except ModuleNotFoundError as e:
        st.error(f"Missing module: {e.name}. Please install scikit-learn.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {repr(e)}")
        st.stop()

loaded_model = load_model()

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

def calculate_risk_score(input_data):
    """Calculate a risk score based on input parameters"""
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope = input_data[:11]
    
    risk_score = 0
    
    # Age risk
    if age > 60: risk_score += 20
    elif age > 50: risk_score += 15
    elif age > 40: risk_score += 10
    
    # Cholesterol risk
    if chol > 240: risk_score += 20
    elif chol > 200: risk_score += 10
    
    # Blood pressure risk
    if trestbps > 140: risk_score += 15
    elif trestbps > 120: risk_score += 8
    
    # Chest pain type
    risk_score += cp * 8
    
    # Exercise angina
    if exang == 1: risk_score += 15
    
    # Oldpeak
    risk_score += oldpeak * 10
    
    # Max heart rate (lower is riskier for older people)
    if age > 50 and thalach < 120: risk_score += 10
    
    return min(risk_score, 100)

def create_gauge_chart(risk_score):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 24, 'color': '#2C3E50', 'family': 'Arial'}},
        number={'font': {'color': '#2C3E50'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1ABC9C", 'tickfont': {'color': '#2C3E50'}},
            'bar': {'color': "#1ABC9C"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#1ABC9C",
            'steps': [
                {'range': [0, 30], 'color': '#E8F5E9'},
                {'range': [30, 60], 'color': '#FFF3CD'},
                {'range': [60, 100], 'color': '#FFEBEE'}
            ],
            'threshold': {
                'line': {'color': "#E74C3C", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='#F5F5F5', font={'color': '#2C3E50'})
    return fig

def create_parameter_radar(input_data):
    """Create radar chart for parameter visualization"""
    categories = ['Age', 'BP', 'Cholesterol', 'Heart Rate', 'Oldpeak']
    age, _, _, trestbps, chol, _, _, thalach, _, oldpeak, _ = input_data[:11]
    
    # Normalize values to 0-100 scale
    values = [
        min((age / 100) * 100, 100),
        min((trestbps / 200) * 100, 100),
        min((chol / 400) * 100, 100),
        min((thalach / 220) * 100, 100),
        min((oldpeak / 6) * 100, 100)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#1ABC9C',
        fillcolor='rgba(26, 188, 156, 0.25)',
        line_width=3
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#16A085', gridwidth=1, tickfont={'color': '#2C3E50'}),
            bgcolor='#FFFFFF',
            angularaxis=dict(gridcolor='#16A085', linecolor='#1ABC9C', tickfont={'color': '#2C3E50'})
        ),
        showlegend=False,
        height=400,
        title={'text': "Parameter Analysis", 'font': {'color': '#2C3E50', 'size': 16, 'family': 'Arial'}},
        paper_bgcolor='#F5F5F5',
        font={'color': '#2C3E50'}
    )
    return fig

def diabetes_prediction(input_data):
    """Make prediction using the loaded model"""
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    probability = loaded_model.predict_proba(input_array) if hasattr(loaded_model, 'predict_proba') else None
    
    return prediction[0], probability

def main():
    # Header
    st.markdown('<div class="main-header">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/cardiogram.png", width=100)
        st.title("Navigation")
        page = st.radio("Go to", ["Prediction", "History", "Information", "About"])
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("Total Predictions", len(st.session_state.prediction_history))
        
    if page == "Prediction":
        show_prediction_page()
    elif page == "History":
        show_history_page()
    elif page == "Information":
        show_information_page()
    elif page == "About":
        show_about_page()

def show_prediction_page():
    st.header("Enter Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic Information")
        age = st.number_input('Age', min_value=1, max_value=120, value=50)
        sex = st.selectbox('Sex', options=[1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female')
        cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3], 
                         format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x])
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], 
                          format_func=lambda x: 'Yes' if x == 1 else 'No')
    
    with col2:
        st.subheader("Clinical Measurements")
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
        chol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
        oldpeak = st.number_input('ST Depression (Oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    with col3:
        st.subheader("ECG & Exercise")
        restecg = st.selectbox('Resting ECG Results', options=[0, 1, 2],
                              format_func=lambda x: ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][x])
        exang = st.selectbox('Exercise Induced Angina', options=[0, 1],
                            format_func=lambda x: 'Yes' if x == 1 else 'No')
        slope = st.selectbox('ST Segment Slope', options=[0, 1, 2],
                            format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
        ca = st.selectbox('Number of Major Vessels (0-3)', options=[0, 1, 2, 3])
        thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3],
                           format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown'][x])
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        predict_button = st.button('🔍 Analyze Heart Disease Risk')
    
    if predict_button:
        input_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        
        with st.spinner('Analyzing patient data...'):
            prediction, probability = diabetes_prediction(input_values)
            risk_score = calculate_risk_score(input_values)
        
        # Save to history
        st.session_state.prediction_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'age': age,
            'prediction': prediction,
            'risk_score': risk_score,
            'data': input_values
        })
        
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Results columns
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            # Use risk_score for display instead of just model prediction
            # Risk Score: 0-30 = Low, 31-60 = Moderate, 61-100 = High
            if risk_score <= 30:
                st.markdown('<div class="info-box risk-low">', unsafe_allow_html=True)
                st.success("### ✅ Low Risk")
                st.write("The analysis indicates a **lower likelihood** of heart disease.")
                st.markdown('</div>', unsafe_allow_html=True)
            elif risk_score <= 60:
                st.markdown('<div class="info-box" style="background-color: #3A3520; border-left: 5px solid #F39C12;">', unsafe_allow_html=True)
                st.warning("### ⚠️ Moderate Risk")
                st.write("The analysis indicates a **moderate likelihood** of heart disease.")
                st.write("**Recommendation:** Please monitor your health and consult with a healthcare provider.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box risk-high">', unsafe_allow_html=True)
                st.error("### ⚠️ High Risk")
                st.write("The analysis indicates a **higher likelihood** of heart disease.")
                st.write("**Recommendation:** Please consult with a cardiologist for further evaluation.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show model prediction separately
            model_result = "High Risk" if prediction == 1 else "Low Risk"
            st.write(f"**Model Prediction:** {model_result}")
            
            if probability is not None:
                st.write(f"**Confidence:** {max(probability[0]) * 100:.1f}%")
        
        with result_col2:
            st.plotly_chart(create_gauge_chart(risk_score), use_container_width=True)
        
        # Parameter visualization
        st.markdown("---")
        st.subheader("Parameter Analysis")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.plotly_chart(create_parameter_radar(input_values), use_container_width=True)
        
        with viz_col2:
            # Create bar chart for key metrics
            metrics_df = pd.DataFrame({
                'Parameter': ['Age', 'Blood Pressure', 'Cholesterol', 'Max Heart Rate'],
                'Value': [age, trestbps, chol, thalach],
                'Normal Range': [50, 120, 200, 150]
            })
            
            fig = px.bar(metrics_df, x='Parameter', y=['Value', 'Normal Range'], 
                        barmode='group', title="Key Metrics Comparison",
                        color_discrete_sequence=['#1ABC9C', '#16A085'])
            fig.update_layout(
                paper_bgcolor='#F5F5F5',
                plot_bgcolor='#FFFFFF',
                font={'color': '#2C3E50', 'family': 'Arial'},
                title={'font': {'color': '#2C3E50', 'size': 16}},
                xaxis={'gridcolor': 'rgba(0, 0, 0, 0.1)', 'tickfont': {'color': '#2C3E50'}},
                yaxis={'gridcolor': 'rgba(0, 0, 0, 0.1)', 'tickfont': {'color': '#2C3E50'}}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Health Recommendations")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.info("**Lifestyle**\n\n✓ Regular exercise\n✓ Balanced diet\n✓ Stress management")
        
        with rec_col2:
            st.info("**Monitoring**\n\n✓ Regular check-ups\n✓ Blood pressure tracking\n✓ Cholesterol monitoring")
        
        with rec_col3:
            st.info("**Prevention**\n\n✓ Quit smoking\n✓ Limit alcohol\n✓ Maintain healthy weight")

def show_history_page():
    st.header("📜 Prediction History")
    
    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet. Go to the Prediction page to analyze patient data.")
    else:
        # Display history as a table
        history_df = pd.DataFrame([
            {
                'Timestamp': h['timestamp'],
                'Age': h['age'],
                'Result': 'High Risk' if h['prediction'] == 1 else 'Low Risk',
                'Risk Score': f"{h['risk_score']:.1f}"
            }
            for h in st.session_state.prediction_history
        ])
        
        st.dataframe(history_df, use_container_width=True)
        
        # Statistics
        st.markdown("---")
        st.subheader("Statistics")
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        high_risk = sum(1 for h in st.session_state.prediction_history if h['prediction'] == 1)
        low_risk = len(st.session_state.prediction_history) - high_risk
        avg_risk = np.mean([h['risk_score'] for h in st.session_state.prediction_history])
        
        stat_col1.metric("High Risk Cases", high_risk)
        stat_col2.metric("Low Risk Cases", low_risk)
        stat_col3.metric("Average Risk Score", f"{avg_risk:.1f}")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

def show_information_page():
    st.header("ℹ️ Understanding Heart Disease")
    
    tab1, tab2, tab3 = st.tabs(["Risk Factors", "Symptoms", "Prevention"])
    
    with tab1:
        st.subheader("Major Risk Factors")
        st.write("""
        - **Age**: Risk increases with age
        - **High Blood Pressure**: Can damage arteries
        - **High Cholesterol**: Can lead to plaque buildup
        - **Smoking**: Damages blood vessels
        - **Diabetes**: Increases risk significantly
        - **Obesity**: Puts extra strain on the heart
        - **Physical Inactivity**: Weakens cardiovascular system
        - **Family History**: Genetic predisposition
        """)
    
    with tab2:
        st.subheader("Common Symptoms")
        st.write("""
        - Chest pain or discomfort (angina)
        - Shortness of breath
        - Pain in neck, jaw, throat, or back
        - Pain or weakness in legs or arms
        - Fatigue
        - Irregular heartbeat
        - Dizziness or lightheadedness
        
        **⚠️ If experiencing severe symptoms, seek immediate medical attention!**
        """)
    
    with tab3:
        st.subheader("Prevention Strategies")
        st.write("""
        1. **Healthy Diet**: Focus on fruits, vegetables, whole grains
        2. **Regular Exercise**: At least 150 minutes per week
        3. **Weight Management**: Maintain healthy BMI
        4. **Quit Smoking**: Single most important step
        5. **Limit Alcohol**: Moderate consumption only
        6. **Manage Stress**: Practice relaxation techniques
        7. **Regular Check-ups**: Monitor blood pressure and cholesterol
        8. **Quality Sleep**: 7-9 hours per night
        """)

def show_about_page():
    st.header("About This System")
    
    st.write("""
    ### Heart Disease Prediction System
    
    This application uses machine learning to assess heart disease risk based on various clinical parameters.
    
    **Developed by:** ARYAN
    
    **Version:** 2.0 Enhanced
    
    #### Features:
    - ✅ ML-based prediction using trained model
    - ✅ Risk score calculation
    - ✅ Interactive visualizations
    - ✅ Prediction history tracking
    - ✅ Educational information
    - ✅ Professional UI/UX
    
    #### Disclaimer:
    This tool is for educational and informational purposes only. It should not replace professional medical advice,
    diagnosis, or treatment. Always consult qualified healthcare providers for medical concerns.
    
    #### Technology Stack:
    - Python
    - Streamlit
    - Scikit-learn
    - Plotly
    - NumPy & Pandas
    """)
    
    st.markdown("---")
    st.info("📧 For questions or feedback, please contact your healthcare provider.")

if __name__ == '__main__':
    main()