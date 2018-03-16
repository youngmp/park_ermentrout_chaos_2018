#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ZetCode Tkinter tutorial

This program shows a simple
menu. It has one action, which
will terminate the program, when
selected. 

Author: Jan Bodnar
Last modified: November 2015
Website: www.zetcode.com
"""

from Tkinter import Tk, Frame, Menu, Checkbutton, Radiobutton
from Tkinter import BooleanVar, BOTH, Label, LEFT, RIGHT, IntVar


class Example(Frame):
  
    def __init__(self, parent):
        Frame.__init__(self, parent)   
         
        self.parent = parent        
        self.initUI()
        
        
    def initUI(self):
      
        self.parent.title("Thetaslowmod GUI")

        self.pack(fill=BOTH, expand=True)
        self.var = BooleanVar()

        # checkboxes for simulation parameters
        cb = Checkbutton(self, text="Run Full Sim",
                         variable=self.var, command=self.onClick(run_full=True))
        cb.select()
        cb.place(x=0, y=0)

        cb2 = Checkbutton(self, text="Run Phase Sim",
                          variable=self.var, command=self.onClick(run_phase=True))
        cb2.select()
        cb2.place(x=0, y=20)

        # radio button here for 

        v = IntVar()
        W=Label(self, 
                text="Choose option:",
                padx=20, justify=LEFT).pack()
        
        Radiobutton(self, 
                    text="Save Last",
                    padx = 20, 
                    variable=v, 
                    value=1).pack(anchor=W)
        Radiobutton(self, 
                    text="Use Last",
                    padx = 20, 
                    variable=v, 
                    value=2).pack(anchor=W)

        
    def onClick(self,run_full=False,run_phase=False):
        pass

    def onExit(self):
        self.quit()


def main():
  
    root = Tk()
    root.geometry("250x150+300+300")
    app = Example(root)
    root.mainloop()  


if __name__ == '__main__':
    main() 
