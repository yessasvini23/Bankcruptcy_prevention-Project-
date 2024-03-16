import pickle
import streamlit as st
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the model
load_file = open('rf_classifier.pkl', 'rb')
model = pickle.load(load_file)
load_file.close()

# Prediction function
def predict(Industrial_Risk, Management_Risk, Financial_Flexibility, Credibility, Competitiveness, Operating_Risk):
    try:
        prediction = model.predict([[Industrial_Risk, Management_Risk, Financial_Flexibility, Credibility, Competitiveness, Operating_Risk]])
        return prediction[0]  # Use prediction[0] to get the actual prediction value
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    st.title('Bankruptcy Prevention Prediction 📉')
    st.markdown('Welcome to the Bankruptcy Prevention Prediction. This is a Random Forest machine learning model to predict BANKRUPTCY or NOT.')
    st.markdown('Use the form below to predict whether a company will go bankrupt or not.')          

    
    Industrial_Risk = st.selectbox('Industrial Risk:', [0, 0.5, 1], key='Industrial_Risk')
    Management_Risk = st.selectbox('Management Risk:', [0, 0.5, 1], key='Management_Risk')
    Financial_Flexibility = st.selectbox('Financial Flexibility:', [0, 0.5, 1], key='Financial_Flexibility')
    Credibility = st.selectbox('Credibility:', [0, 0.5, 1], key='Credibility')
    Competitiveness = st.selectbox('Competitiveness:', [0, 0.5, 1], key='Competitiveness')
    Operating_Risk = st.selectbox('Operating Risk:', [0, 0.5, 1], key='Operating_Risk')


    if st.button('Predict'):
        Result = predict(Industrial_Risk, Management_Risk, Financial_Flexibility, Credibility, Competitiveness, Operating_Risk)
        if Result == 0:
            st.markdown('<h1 style="color: red; font-size: 36px;">Prediction: Bankruptcy</h1>', unsafe_allow_html=True)
        else:
            st.markdown('<h1 style="color: green; font-size: 36px;">Prediction: Non-Bankruptcy</h1>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
