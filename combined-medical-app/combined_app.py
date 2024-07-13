import streamlit as st
import pyttsx3
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize text-to-speech engine
engine = pyttsx3.init()

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

# Load trained model for Parkinson's disease detection
parkinson_model = joblib.load(r"C:\sudhanshu_projects\project-task-training-course\Parkinson-disease-detection\Parkinson-disease-detection.pkl")

# Load trained model for diabetes disease detection
diabetes_model = joblib.load(r"C:\sudhanshu_projects\project-task-training-course\diabetes-prediction\diabetics_prediction.pkl")

# Load trained model for Heart disease detection
heart_model = joblib.load(r"C:\sudhanshu_projects\project-task-training-course\CardicArrest-prediction\heart_failure_prediction.pkl")

# Load data
@st.cache_data
def load_data():
    diabetic_df = pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\diabetes_prediction_dataset.csv")
    return diabetic_df

# Preprocess data
def preprocess_data(diabetic_df):
    # Label encoding
    lr = LabelEncoder()
    col = ["gender", "smoking_history"]
    for i in col:
        diabetic_df[i] = lr.fit_transform(diabetic_df[i])
    
    X = diabetic_df.iloc[:, :-1].values
    y = diabetic_df.iloc[:, -1].values
    
    # Standardization
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    
    return X, y, X_sc

# Train models
def train_models(X, y, X_sc):
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_sc, y)
        y_pred = model.predict(X_sc)
        accuracy = accuracy_score(y, y_pred)
        results[name] = accuracy
    
    return results

# Visualizations
def plot_visualizations(diabetic_df):
    st.subheader("Visualizations")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.countplot(data=diabetic_df, x="diabetes", ax=axs[0, 0])
    axs[0, 0].set_title("Diabetes Count")
    
    sns.barplot(data=diabetic_df, x="gender", y="diabetes", ax=axs[0, 1])
    axs[0, 1].set_title("Gender vs Diabetes")
    
    sns.barplot(data=diabetic_df, x="hypertension", y="diabetes", ax=axs[1, 0])
    axs[1, 0].set_title("Hypertension vs Diabetes")
    
    sns.barplot(data=diabetic_df, x="heart_disease", y="diabetes", ax=axs[1, 1])
    axs[1, 1].set_title("Heart Disease vs Diabetes")
    
    st.pyplot(fig)

# Define prediction functions
def predict_diabetes(input_data):
    diabetic_df = load_data()
    le=LabelEncoder()
    
    col=["gender","smoking_history"]
    for i in col:
      diabetic_df[i]=le.fit_transform(diabetic_df[i])
 
    
    x=diabetic_df.drop("diabetes",axis=1)
    y=diabetic_df["diabetes"]
     
    model = RandomForestClassifier()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    
    model.fit(x_train,y_train)
    input_df = pd.DataFrame([input_data])
    return model.predict(input_data)[0]
    
    # diabetic_df = load_data()
    # X, y, X_sc = preprocess_data(diabetic_df)
    # model = RandomForestClassifier()
    # model.fit(X_sc, y)
    # input_data_sc = StandardScaler().fit_transform(input_data)
    # return model.predict(input_data_sc)[0]

def predict_heart_attack_death(input_data):
    diabetic_df = pd.read_csv(r"C:\sudhanshu_projects\project-task-training-course\CardicArrest-prediction\heart_failure_dataset.csv")
    
    x=diabetic_df.drop(["time","DEATH_EVENT"],axis=1)
    y=diabetic_df["DEATH_EVENT"]
     
    model = LogisticRegression()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

    #Now here we do scaling.
    sc=StandardScaler()
    x_train_sc=sc.fit_transform(x_train)
    x_test_sc=sc.transform(x_test) 
    
    model.fit(x_train_sc,y_train)
    
    input_data_sc=sc.fit_transform(input_data)
    return model.predict(input_data_sc)[0]
      
    # input_df_heart = pd.DataFrame([input_data])
    #Here we do label encoding
    # le=LabelEncoder()
    # for col in input_df_heart.columns:
    #    input_df_heart[col]=le.fit_transform(input_df_heart[col])
    #Now here we do scaling.
    # sc=StandardScaler()
    # input_df_heart=sc.fit_transform(input_df_heart)    
    
    # return int(diabetes_model.predict(input_df_heart)[0])

def predict_parkinson(input_data):
    input_df_parkinson = pd.DataFrame([input_data]) 
    return int(parkinson_model.predict(input_df_parkinson)[0])

# Streamlit App
def main():
    st.title("Health Condition Prediction App")

    option = st.selectbox(
        "Select the prediction model",
        ("Diabetes Prediction", "Heart Attack Death Prediction", "Parkinson's Disease Detection")
    )

    if option == "Diabetes Prediction":
        st.subheader("Diabetes Prediction")
        
        # gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level
        # Add input fields for diabetes prediction
        gender = st.radio("Gender", [0,1])
        age = st.slider("Age", 0, 100, 1)
        hypertension=st.radio("Hypertension",[0,1])
        heart_disease=st.radio("Heart_Disease",[0,1])
        smoking_history=st.radio("Smoking_History",[0,1,2,3])
        bmi=st.slider("Bmi",10.0,10.0,60.0)
        HbA1c_level=st.slider("HbA1c_level",0.0,10.0,0.0)
        blood_glucose_level=st.slider("Blood_Glucose_Level",0,500,50)
       

        if st.button("Predict Diabetes"):    
            input_data = [[gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level]]
            result = predict_diabetes(input_data)
            if result == 1:
                st.warning("The person is diagnosed with diabetes.")
                tips = ["Eat healthy foods", "Be more physically active", "Lose excess weight", "Monitor your blood sugar", "Take your medication as prescribed", "Manage stress"]
                st.write("Here are some tips to manage diabetes:")
                for tip in tips:
                    st.write("- " + tip)
                text_to_speech("The person is diagnosed with diabetes. Here are some tips to manage diabetes: " + ", ".join(tips))
            else:
                st.success("The person is not diagnosed with diabetes.")
                text_to_speech("The person is not diagnosed with diabetes. so don't take any stress any enjoy your life.")
                
                
    elif option == "Heart Attack Death Prediction":
        st.subheader("Heart Attack Death Prediction")
        
        #age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking
        # Add input fields for heart attack death prediction
        age = st.slider("Age", 0,100,1)
        anaemia=st.selectbox("Anemia",[0,1])
        creatinine_phosphokinase=st.slider("Creatinine_Phosphokinase",50,8000,50)
        diabetes=st.selectbox("Diabetes",[0,1])
        ejection_fraction= st.slider("Ejection_fraction", 15,100,15)
        high_blood_pressure=st.selectbox("High_Blood_Pressure",[0,1])
        platelets=st.slider("Platelets",1000000,10000000,1000000)
        serum_creatinine=st.slider("Serum_creatinine", 0.0,10.0,0.1)
        serum_sodium=st.slider("Serium_Sodium",100,150,100)
        sex = st.selectbox("Sex", [0,1])
        smoking = st.selectbox("Smoking", [0,1])
       
        if st.button("Predict Heart Attack Death"):
            input_data = [[age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking]]
            result = predict_heart_attack_death(input_data)
            if result == 1:
                st.warning("The person is at risk of heart attack death.")
                tips = ["Quit smoking", "Eat a healthy diet", "Exercise regularly", "Maintain a healthy weight", "Manage stress", "Monitor your blood pressure and cholesterol"]
                st.write("Here are some tips to reduce the risk of heart attack:")
                for tip in tips:
                    st.write("- " + tip)
                text_to_speech("The person is at risk of heart attack death. Here are some tips to reduce the risk of heart attack: " + ", ".join(tips))
            else:
                st.success("The person is not at risk of heart attack death.")
                text_to_speech("You does not have any heart related problem. so don't take any stress any enjoy your life.")

    elif option == "Parkinson's Disease Detection":
        st.subheader("Parkinson's Disease Detection")
        
        #MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE
        # Add input fields for Parkinson's disease detection
        mdvp_fo = st.number_input("MDVP:Fo(Hz)", 0.0)
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", 0.0)
        mdvp_flo = st.number_input("MDVP:Flo(Hz)", 0.0)
        mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0)
        mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0)
        mdvp_rap = st.number_input("MDVP:RAP", 0.0)
        mdvp_ppq = st.number_input("MDVP:PPQ", 0.0)
        jitter_ddp = st.number_input("Jitter:DDP", 0.0)
        mdvp_shimmer = st.number_input("MDVP:Shimmer", 0.0)
        mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0)
        shimmer_apq3 = st.number_input("Shimmer:APQ3", 0.0)
        shimmer_apq5 = st.number_input("Shimmer:APQ5", 0.0)
        mdvp_apq = st.number_input("MDVP:APQ", 0.0)
        shimmer_dda = st.number_input("Shimmer:DDA", 0.0)
        nhr = st.number_input("NHR", 0.0)
        hnr = st.number_input("HNR", 0.0)
        rpde = st.number_input("RPDE", 0.0)
        dfa = st.number_input("DFA", 0.0)
        spread1 = st.number_input("Spread1", 0.0)
        spread2 = st.number_input("Spread2", 0.0)
        d2 = st.number_input("D2", 0.0)
        ppe = st.number_input("PPE", 0.0)

        if st.button("Predict Parkinson's Disease"):
            input_data = [mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs, mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
            result = predict_parkinson(input_data)
            if result == 1:
                st.warning("The person is diagnosed with Parkinson's disease.")
                tips = ["Get regular exercise", "Eat a balanced diet", "Stay socially active", "Get enough sleep", "Take medications as prescribed", "See your healthcare provider regularly"]
                st.write("Here are some tips to manage Parkinson's disease:")
                for tip in tips:
                    st.write("- " + tip)
                text_to_speech("The person is diagnosed with Parkinson's disease. Here are some tips to manage Parkinson's disease: " + ", ".join(tips))
            else:
                st.success("The person is not diagnosed with Parkinson's disease.")
                text_to_speech("You does not have any parkinson disease. so don't take any stress any enjoy your life.")



if __name__ == "__main__":
    main()
