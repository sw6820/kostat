import pandas as pd
from logger import get_logger

class Preprocess:
    def __init__(self):
        self.logger = get_logger("preprocess")

    def train_preprocess(self, train: pd.DataFrame) -> pd.DataFrame:
        '''
        train data를 전처리합니다.
        '''

        try:
            # digit은 classification을 위한 것이므로 
            # 연속형 변수 dtype인 int, float를 범주형 변수를 위한 dtype인 str으로 변경
            train.loc[:,'digit_2'] = train.loc[:,'digit_2'].astype(str)
            train.loc[:,'digit_3'] = train.loc[:,'digit_3'].astype(str)
            
            # NaN은 덧셈이 불가능하므로 빈 문자열로 교체
            train = train.fillna('')

            # 변수 합치기
            train.loc[:,"label"] = train["digit_3"]
            train.loc[:,"text"] = train["text_obj"].apply(lambda x: x + " ") + train["text_mthd"].apply(lambda x: x + " ") + train["text_deal"]
            train['text'] = train['text'].str.replace("[^a-zA-Z가-힣 ]","", regex=True)
            
            self.logger.info("Success train data Preprocessing")
        except:
            self.logger.exception("Fail train data Preprocessing")

        return train


    def test_preprocess(self, test: pd.DataFrame) -> pd.DataFrame:
        '''
        test data를 전처리합니다.
        '''
        try:
            test.loc[:,'digit_1'] = test.loc[:,'digit_1'].astype(str)
            test.loc[:,'digit_2'] = test.loc[:,'digit_2'].astype(str)
            test.loc[:,'digit_3'] = test.loc[:,'digit_3'].astype(str)
            test = test.fillna('')
            test.loc[:,"text"] = test["text_obj"].apply(lambda x: x + " ") + test["text_mthd"].apply(lambda x: x + " ") + test["text_deal"]
            test['text'] = test['text'].str.replace("[^a-zA-Z가-힣 ]","", regex=True)

            self.logger.info("Success test data Preprocessing")
        except:
            self.logger.exception("Fail test data Preprocessing")

        return test