# Neural-Net-for-Default-of-Credit-Card-Clients
 - Implemented Neural Network code from ground up in python
 	- To understand hyper parameters
 	- To understand math involved in NN
 - Implemented a Cross Validation Function for Neural Netwrok but is generic in terms of choosing parameters
 	- Helps in understanding how differnt paramters help and change the accuracy
 - Code is generic and can be used to create DNN but it is not advised to do so as this is not parallel
 - Implemented both Linear model and Non linear model for NN
 - Impemented for both Classification data and Regression data.


## Data Set Used:
<b>Default of credit card clients</b><br>
Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#

### Data Description:

There are 25 variables:

1. <b>ID:</b> ID of each client  
2. <b>LIMIT_BAL:</b> Amount of given credit in NT dollars (includes individual and family/supplementary credit  
3. <b>SEX:</b> Gender (1=male, 2=female)  
4. <b>EDUCATION:</b> (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)  
5. <b>MARRIAGE:</b> Marital status (1=married, 2=single, 3=others)  
6. <b>AGE:</b> Age in years  
7. <b>PAY_0:</b> Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
8. <b>PAY_2:</b> Repayment status in August, 2005 (scale same as above)  
9. <b>PAY_3:</b> Repayment status in July, 2005 (scale same as above)  
10. <b>PAY_4:</b> Repayment status in June, 2005 (scale same as above)  
11. <b>PAY_5:</b> Repayment status in May, 2005 (scale same as above)  
12. <b>PAY_6:</b> Repayment status in April, 2005 (scale same as above)  
13. <b>BILL_AMT1:</b> Amount of bill statement in September, 2005 (NT dollar)  
14. <b>BILL_AMT2:</b> Amount of bill statement in August, 2005 (NT dollar)  
15. <b>BILL_AMT3:</b> Amount of bill statement in July, 2005 (NT dollar)  
16. <b>BILL_AMT4:</b> Amount of bill statement in June, 2005 (NT dollar)  
17. <b>BILL_AMT5:</b> Amount of bill statement in May, 2005 (NT dollar)  
18. <b>BILL_AMT6:</b> Amount of bill statement in April, 2005 (NT dollar)  
19. <b>PAY_AMT1:</b> Amount of previous payment in September, 2005 (NT dollar)  
20. <b>PAY_AMT2:</b> Amount of previous payment in August, 2005 (NT dollar)  
21. <b>PAY_AMT3:</b> Amount of previous payment in July, 2005 (NT dollar)  
22. <b>PAY_AMT4:</b> Amount of previous payment in June, 2005 (NT dollar)  
23. <b>PAY_AMT5:</b> Amount of previous payment in May, 2005 (NT dollar)  
24. <b>PAY_AMT6:</b> Amount of previous payment in April, 2005 (NT dollar)  
25. <b>default.payment.next.month:</b> Default payment (1=yes, 0=no)  

## Neural Network:
- Created a NN in python and used it for classification
- Also, used tensorflow to test many different structures each have more than 5 layers