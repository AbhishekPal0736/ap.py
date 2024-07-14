import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
st.title(":globe_with_meridians::MACHINE_LEARNING_PROJECT")
IRIS=pd.read_csv('IRIS.csv')
X=IRIS.drop('species',axis=1)
y=IRIS['species']
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=1234)

st.sidebar.header("Enter New Data")
sepal_length = st.sidebar.number_input("Sepal Length:")
sepal_width =  st.sidebar.number_input("Sepal Width:")
petal_length = st.sidebar.number_input("Petal Length:")
petal_width = st.sidebar.number_input("Petal Width")

classifier=SVC()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
st.write("Accuracy:",accuracy)
st.write("classification_report:")
st.write(report)
#st.button("Make Prediction")
new_data=[[sepal_length,sepal_width,petal_length,petal_width]]
prediction=classifier.predict(new_data)
st.write("prediction:",prediction[0])

st.header("Analaysis the Line Chart")
st.line_chart(IRIS,height=600,width=150)
st.header("Analaysis the Scatter Chart")
st.scatter_chart(IRIS,height=600,width=150)
st.button("Make Prediction")
