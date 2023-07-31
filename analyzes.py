# Import necessary libraries
import streamlit as st # For build GUI 
import base64 # For function backgroumd 


# create dict contians all tests
tests_dict= {
    'Complete Blood Count': {
        'Haemaglobin (Male)': (13, 17), 'Haemaglobin (Female)': (12, 16),
        'Red Blood Cell Count (RBC) (Male)': (4.5, 6.5), 'Red Blood Cell Count (RBC) (Male)': (3.8, 5.8), 
        'Hematocrit (Male)': (40, 54), 'Hematocrit (Female)': (36, 48),
        'Mean Corpuscular Volume (MCV)': (80, 96),
        'Mean Corpuscular Hemoglobin (MCH)': (26, 32),
        'Mean Corpuscular Hemoglobin Concentration (MCHC)': (31, 37), 
        'Red Cell Distribution Width (RDW-CV)': (11.5, 14.5), 
        'Platelet Count': (150, 450),
        'Mean Platelet Volume (MPV)': (7.4, 12),
        'White Blood Cell Count (WBC)': (4, 11),
        'Basophils': ((0.00, 0.1) , (0, 1)),
        'Eosinophils': ((0.00, 0.5) , (0, 5)),
        'Neutrophils': ((2, 7.8) , (50, 70)),
        'Lymphocytes': ((0.6, 4.1) , (20, 45)),
        'Monocytes': ((0.1, 1.8) , (2, 10))
    },
    
    'Liver Functions Test': {
        'S.GPT (ALT)': (0, 40),
        'S.GOT (AST)': (0, 35),
        'Total bilirubin': (0, 1.0),
        'Direct bilirubin': (0, 0.25),
    },
    
    'HAEMATOLOGY': {
        'ESR (1st h.)': (0, 7),
        'ESR (2nd h.)': (0, 14),
    },
    
    'IMMUNOLOGY': {
        'CRP': (0, 6),
        'Thyroid Anti-Microsmal Ab': (0, 100),        
    },
    
    'Thyroid hormones (Young)': {
        'Free T4': (0.8, 2),
        'Free T3': (2.5, 5.9),
        'TSH': (0.27, 5.2),
    },
    
    'Kidney Functions': {
        'Serum Uric Acid': (2.5, 5.7),
    },
    
    'Blood Diseases': {
        'Ferritin': (15, 120),
    },
    
    'CHEMISTRY': {
        'HOMA2 - IR': (0, 1.8),
    },
    
    'Thyroid hormones (Children)': {
        'Free T4': (0.93, 1.7),
        'Free T3': (2, 4.4),
        'TSH': (0.7, 6.4),
    },
}

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    

def check_test_result(category, test_name, value):
    normal_range_min, normal_range_max = tests_dict[category][test_name] # (min, max)
    
    result = 0 if normal_range_max >= value >= normal_range_min else 1
    result_text = 'negative' if result == 0 else 'positive'
    
    message = "Please consult a doctor." if result == 1 else "great your test is good."
    return f"<span style='color:red;'><strong>Test: {test_name}</strong></span>\n\n<span style='color:red;'><strong>Value: {value}</strong></span>\n\n<span style='color:red;'><strong>Result: {result_text}</strong></span>\n\n<span style='color:red;'><strong>Test: {message}</strong></span>"

print(check_test_result('Thyroid hormones (Children)', 'TSH', 20))


# Streamlit app
# set_background('bg3.jpeg')
st.title("Medical Test Analyzer")

category = st.selectbox("Select Test Category", list(tests_dict.keys()))
test_name = st.selectbox("Select Test Name", list(tests_dict[category].keys()))
value = st.number_input("Enter Test Value", value=0.0)

if st.button("Analyze Test"):
    result = check_test_result(category, test_name, value)
    # st.write(result)
    st.markdown(result, unsafe_allow_html=True)
