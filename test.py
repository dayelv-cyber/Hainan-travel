from sklearn import svm
#[0,0]属于0类,[1,1]属于1类
x=[[0,0],[1,1]]
y=[0,1]
#造一个空的 SVM 分类器(还没学任何东西)
ss=svm.SVC()
#照着规律学
ss.fit(x,y)
#预测这个结果是多少
result=ss.predict([[2,2]])
print(result)