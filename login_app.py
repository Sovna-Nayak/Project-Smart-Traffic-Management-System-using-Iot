from db import get_db
import tkinter.messagebox

def getThere(self):
    username = self.name1.get()
    pwd = self.pwd1.get()

    db = get_db()
    users_collection = db['users']

    # Find user with matching username and password
    user = users_collection.find_one({"username": username, "password": pwd})

    if user:
        tkinter.messagebox.showinfo("Login Success", "Welcome to the Traffic Management System")
        there = Traffic()
    else:
        tkinter.messagebox.showerror("Login Failed", "Invalid Username or Password")
