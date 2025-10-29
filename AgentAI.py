import subprocess 

def reactive_agent(prompt): 
    r = subprocess.run(["ollama","run","mistral",prompt],capture_output=True,text=True) 
    return r.stdout 
while True: 
    q=input("Vous : ") 
    if q.lower() in ["quit","exit"]:break 
    print("Agent :",reactive_agent(q))