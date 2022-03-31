import pandas as pd
from logger import get_logger

def dataframe_preprocess(train: pd.DataFrame, test: pd.DataFrame):
    '''
    train, test dataframe을 전처리합니다.
    '''

    logger = get_logger("preprocess")

    try:
        # digit은 classification을 위한 것이므로 
        # 연속형 변수 dtype인 int, float를 범주형 변수를 위한 dtype인 str으로 변경
        train.loc[:,'digit_2'] = train.loc[:,'digit_2'].astype(str)
        train.loc[:,'digit_3'] = train.loc[:,'digit_3'].astype(str)
        test.loc[:,'digit_1'] = test.loc[:,'digit_1'].astype(str)
        test.loc[:,'digit_2'] = test.loc[:,'digit_2'].astype(str)
        test.loc[:,'digit_3'] = test.loc[:,'digit_3'].astype(str)

        # NaN은 덧셈이 불가능하므로 빈 문자열로 교체
        train = train.fillna('')
        test = test.fillna('')

        # 변수 합치기
        train.loc[:,"label"] = train["digit_3"]
        train.loc[:,"text"] = train["text_obj"].apply(lambda x: x + " ") + train["text_mthd"].apply(lambda x: x + " ") + train["text_deal"]
        test.loc[:,"text"] = test["text_obj"].apply(lambda x: x + " ") + test["text_mthd"].apply(lambda x: x + " ") + test["text_deal"]

        columns = ["AI_id", "text", "label"]
        train = train.loc[:,columns]

        train['text'] = train['text'].str.replace("[^a-zA-Z가-힣 ]","", regex=True)
        test['text'] = test['text'].str.replace("[^a-zA-Z가-힣 ]","", regex=True)

        logger.debug("Success Preprocessing")
    except:
        logger.exception("Fail Preprocessing")

    return train, test