import pandas as pd

def summarize_results(results):
    df = pd.DataFrame(results)
    print(df.mean())
    return df
