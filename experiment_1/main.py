from langchain.llms import OpenAI, OpenAIChat
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
import json
import os
import warnings

# 自定义忽略特定的警告信息
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*langchain.*")


os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"

# 连接到大模型
llm_completion = OpenAI(model_name="Qwen1.5-14B")
llm_chat = OpenAIChat(model_name="Qwen1.5-14B")

# 连接到嵌入模型
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# 连接到数据库
db = Milvus(embedding_function=embedding, collection_name="arXiv", connection_args={"host": "172.29.4.47", "port": "19530"})


def optimiz_question(question):
    # 使用大模型对用户输入进行润色（可调整参数）
    opt_question = llm_chat(f"请改写以下查询以便更好地搜索相关文档：{question}")
    return opt_question

# 可调整参数 
def search_arxiv(question, papers_per_query = 5):
    # 搜索相关的论文摘要
    results = db.similarity_search(query=question, k=papers_per_query)
    
    return results

def gen_prompt(question, relevant_papers):
    # 整理从数据库中检索到的信息
    abstracts = "\n".join([paper.page_content for paper in relevant_papers])
    
    # 构造Prompt（可调整参数）
    prompt = f"问题: {question}\n\n相关论文摘要:\n{abstracts}\n\n请根据以上信息回答问题。"
    
    return prompt

def gen_answer(prompt):
    # 使用大模型生成回答
    answer = llm_chat(prompt)
    
    return answer

def ask_arxiv_question(question):
    opt_question = optimiz_question(question)
    relevant_papers = search_arxiv(opt_question)
    prompt = gen_prompt(opt_question, relevant_papers)
    answer = gen_answer(prompt)
    
    
    # 可调整参数
    sources = [{"title": paper.metadata["title"], 
                "authors": paper.metadata["authors"], 
                "url": f"https://arxiv.org/abs/{paper.metadata['access_id']}"} 
               for paper in relevant_papers]
    
    return {"opt_question": opt_question, "answer": answer, "sources": sources}


def answer_of_file():
    with open("./作业1/code/questions.json", 'r', encoding='utf-8') as f:
        questions = json.load(f)
        
    answers = []
    
    for question_data in questions:
        question = question_data["question"]
        response = ask_arxiv_question(question)
        question_data["answer"] = response["answer"]
        answers.append(question_data)
    
    # 保存答案到answer.json文件
    with open("./作业1/code/answer.json", "w", encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

def interactive_terminal(max_iterations=3):
    print("欢迎使用arXiv知识问答系统！输入 'exit' 退出。")
    
    while True:
        question = input("请输入您的问题: ")
        if question.lower() == 'exit':
            print("感谢使用，再见！")
            break
        
        response = ask_arxiv_question(question)
        print(f"大模型优化问题：{response['opt_question']}")
        print(f"答案: {response['answer']}")
        print("相关文献来源:")
        for source in response["sources"]:
            print(f"- {source['title']} by {source['authors']} ({source['url']})")
        print("\n")

if __name__ == "__main__":
    answer_of_file()
    # interactive_terminal()