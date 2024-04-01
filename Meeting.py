import streamlit as st
import importlib.util
from streamlit_option_menu import option_menu


import Meeting_Script_Summarizer
import SentiAnalysis
import WhoSaidWhat
import QnAwithMeetScript
# import PDF_GPT
# import Home

# Define a function to import and run a page script dynamically
def run_page(page_name):
    spec = importlib.util.spec_from_file_location(page_name, f"pages/{page_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

# Configure Streamlit page
#st.set_page_config(
 #   page_title="TA Chatbot",
  #  page_icon="🤖",
#)

# Define page names and their corresponding display names
def app():
    st.write("Meeting Notes Summarizer")

    app = option_menu(
        options=['QnA','Meet Analyser', 'SentiAnalysis','Who Said What'],
        menu_title="Select the menu below :",
        icons=['wechat', 'filetype-pdf','megaphone'],
        default_index=0, 
        orientation='horizontal', # Set default index to 0 (Chatbot)
        styles={
            "container": {"padding": "5!important", "background-color": 'black'},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "15px", "text-align": "left", "margin": "0px",
                         "--hover-color": "blue"},
            "nav-link-selected": {"background-color": "#FE6771","font-size": "12px"},
            "options":{"color":"black"},
        }
    )

    if app == 'Meet Analyser':
        Meeting_Script_Summarizer.app()
    if app == 'SentiAnalysis':
        SentiAnalysis.app()
    if app == 'QnA':
        QnAwithMeetScript.app()
    elif app == 'Who Said What':
        WhoSaidWhat.app()

if __name__ == '__main__':
  app()
