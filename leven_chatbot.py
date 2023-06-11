import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 챗봇 클래스를 정의
class Levenstein_ChatBot:
    # 챗봇 객체를 초기화하는 메서드, 초기화 시에는 입력된 데이터 파일을 로드하고, TfidfVectorizer를 사용해 질문 데이터를 벡터화함
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    # CSV 파일로부터 질문과 답변 데이터를 불러오는 메서드
    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()
        answers = data['A'].tolist()
        return questions, answers


    # 레벤슈타인 거리로 계산한 유사도로 질문 답변 반환
    def find_best_answer_leven(a, b):
        ''' 레벤슈타인 거리 계산하기 '''
        if a == b: return 0  # 같으면 0을 반환
        a_len = len(a)  # a 길이
        b_len = len(b)  # b 길이
        if a == "": return b_len
        if b == "": return a_len

        matrix = [[] for i in range(a_len + 1)]  # 리스트 컴프리헨션을 사용하여 1차원 초기화
        for i in range(a_len + 1):  # 0으로 초기화
            matrix[i] = [0 for j in range(b_len + 1)]  # 리스트 컴프리헨션을 사용하여 2차원 초기화
        # 0일 때 초깃값을 설정
        for i in range(a_len + 1):
            matrix[i][0] = i
        for j in range(b_len + 1):
            matrix[0][j] = j
        # 표 채우기 --- (※2)
        # print(matrix,'----------')
        for i in range(1, a_len + 1):
            ac = a[i - 1]
            # print(ac,'=============')
            for j in range(1, b_len + 1):
                bc = b[j - 1]
                # print(bc)
                cost = 0 if (ac == bc) else 1  # 파이썬 조건 표현식 예:) result = value1 if condition else value2
                matrix[i][j] = min([
                    matrix[i - 1][j] + 1,  # 문자 제거: 위쪽에서 +1
                    matrix[i][j - 1] + 1,  # 문자 삽입: 왼쪽 수에서 +1
                    matrix[i - 1][j - 1] + cost  # 문자 변경: 대각선에서 +1, 문자가 동일하면 대각선 숫자 복사
                ])

        return matrix[a_len][b_len]



# 데이터 파일의 경로를 지정합니다.
filepath = 'ChatbotData.csv'

# 챗봇 객체를 생성합니다.
chatbot = Levenstein_ChatBot(filepath)

# 학습 질문데이터 설정
data = pd.read_csv(filepath)
q_list = data['Q'].to_list()

# '종료'라는 입력이 나올 때까지 사용자의 입력에 따라 챗봇의 응답을 출력하는 무한 루프를 실행합니다.
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    distance = [chatbot.find_best_answer_leven(question, input_sentence) for question in q_list]
    best_match_index = similarities.index(min(distance))
    print("Chatbot:", chatbot.answers[best_match_index])
    
