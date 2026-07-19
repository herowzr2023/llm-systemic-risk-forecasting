import pandas as pd

df = pd.read_csv("combined_data_for_工商银行_final_data.csv")
df.工商银行_CoVaR = df.工商银行_CoVaR * 100
df.to_csv("combined_data_for_工商银行_final_data_mult.csv", index=False)
print(df)
