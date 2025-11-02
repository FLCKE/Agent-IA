import subprocess 

def reactive_agent(prompt): 
    r = subprocess.run(["ollama","run","mistral",prompt],capture_output=True,text=True) 
    return r.stdout 
user_name = None 
nameValid = False
while True: 
    msg = input("Vous : ") 
    if "je m'appelle" in msg.lower() or user_name != None: 
        user_name = msg.split()[-1] 
        if nameValid==False:
            print(f"Enchanté {user_name} !") 
            nameValid = True
        if msg.lower() in ["quit","exit"]:break 
        print("Agent :",reactive_agent(msg))
    elif "mon nom" in msg.lower() and user_name: 
        print(f"Tu t'appelles {user_name}.") 
    else: 
        print("Je ne me souviens pas, désolé.")
    

    