import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
#from pickle import dump
from pickle import load

# giving a title
st.title('Bankruptcy Predictions')


def user_input_features():
    industrial_risk = st.selectbox('Industrial Risk',['0.0','0.5','1.0'])
    management_risk = st.selectbox('Management Risk',['0.0','0.5','1.0'])
    financial_flexibility = st.selectbox('Financial Flexibility',['0.0','0.5','1.0'])
    credibility = st.selectbox('Credibility',['0.0','0.5','1.0'])
    competitiveness = st.selectbox('Competitiveness',['0.0','0.5','1.0'])
    operating_risk = st.selectbox('Operating Risk',['0.0','0.5','1.0'])
    model = st.selectbox('Model',['Logistic Regression','Naive Bayes Classifier','K- Nearest Neighbors Classifier','SVC','Decision Tree Classifier','Bagging Classifier','Random Forest Classifier'])

    data = {'industrial_risk':industrial_risk,
            'management_risk':management_risk,
            'financial_flexibility':financial_flexibility,
            'credibility':credibility,
            'competitiveness':competitiveness,
            'operating_risk':operating_risk}

    features = pd.DataFrame(data,index = [0])
    return (features,model)

def main():

    # # title
    # html_temp = '''
    # <div style='background-color:blue'>
    # <h2 style='color:white;text-align:center'>Bankruptcy Predictions</h2>
    # '''
    # st.markdown(html_temp,unsafe_allow_html=True)
    
    # getting data from user
    df,model = user_input_features()
    loaded_model = load(open('rf_classifier.pkl','rb'))

    # Creating a button for prediction
    if st.button('Predict'):
        if model == 'Logistic Regression':
            predction = loaded_model[0].predict(df)
            prediction_proba = loaded_model[0].predict_proba(df)
        elif model == 'Naive Bayes Classifier':
            predction = loaded_model[1].predict(df)
            prediction_proba = loaded_model[1].predict_proba(df)
        elif model == 'K- Nearest Neighbors Classifier':
            predction = loaded_model[2].predict(df)
            prediction_proba = loaded_model[2].predict_proba(df)
        elif model == 'SVC':
            predction = loaded_model[3].predict(df)
            prediction_proba = loaded_model[3].predict_proba(df)
        elif model == 'Decision Tree Classifier':
            predction = loaded_model[4].predict(df)
            prediction_proba = loaded_model[4].predict_proba(df)
        elif model == 'Bagging Classifier':
            predction = loaded_model[5].predict(df)
            prediction_proba = loaded_model[5].predict_proba(df)
        else:    
            predction = loaded_model[6].predict(df)
            prediction_proba = loaded_model[6].predict_proba(df)

        st.subheader('Predicted Result')
        st.success('Non-bankruptcy' if prediction_proba[0][1]>prediction_proba[0][0] else 'Bankruptcy')

        # st.subheader('Prediciton Probability')
        # st.info(prediction_proba)

if __name__ == '__main__':
    main()
