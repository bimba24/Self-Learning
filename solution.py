import quadraticEquation


def solution(typeOfEqu,equation):
    if(typeOfEqu =="Quadratic"):
        answer=quadraticEquation.dataMain(equation)
    else:
        answer="None"
    return answer