# program takes an input file with tabular data and determines the metadata for each column. 
# The meta data will cover data type specific to the input database technology specified along with sizing details.
# Examples of sizing details are: character length, precision, scale


# to reference environment variables
from dotenv import load_dotenv
import os
# to work with imported csv file
import pandas as pd
# to use chat model
from langchain.chat_models import AzureChatOpenAI
# to define roles in chat
from langchain.schema import HumanMessage, SystemMessage, AIMessage
# to create prompt template
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
# set up langchain
from langchain.chains import LLMChain
# to get token and cost per datatype determination for a csv file
from langchain.callbacks import get_openai_callback
# for UX
import streamlit as st



def main():

    load_dotenv()

    st.set_page_config(page_title='Metadata determination', layout='wide')
    st.header('Given a delimited file, determine datatypes for the columns of data ')
    
    with st.expander('More details ...'):
        
        st.text('''
                Upload a delimited file. Ensure that it has column headers in the 1st row. The column data 
                is passed to an LLM and the response is the suggested data type based on the column data
                based on the database technology specified.
                If the data for a column is spare, then response indicates that the datatype cannot
                be determined.
                The token metrics and cost associated to determine metadata for a delimited file shows 
                after processing is complete
    ''')
    
    

    input_file_abs = st.file_uploader('Upload file', type = ['csv'])

    if input_file_abs != None:
        input_df = pd.read_csv(input_file_abs, encoding='unicode_escape')

    # db technology input
    dbtech = st.text_input('Database technology', value="")

   
    button1 = st.button('Process')

    

    # output dataframe initialization
    outdf = pd.DataFrame()

    if button1:
        outdf, total_token_count, prompt_tokens_count, completion_tokens_count, cost = LLMprocess(input_df, dbtech)

        st.header('Final response')
        st.dataframe(data=outdf, use_container_width=True)

        
        
        
        




#########################################################################################################

def LLMprocess(df, dbt):
    '''
    Takes a dataframe. Pass dataframe column one at a time to LLM chat model 
    to determine datatype based on database technology
    
    Args:
    df - input dataframe to pass to LLM
    dbt - database technology based on which datatypes are to be determined for the data frame columns

    Returns:
    dataframe of column names and suggested datatypes
    token metrics and cost for processing

    '''

    # set up chat model
    chat_model = AzureChatOpenAI(deployment_name = 'gpt-4')

    # create system prompt template
    sys_prompt_temp = PromptTemplate.from_template(
    template = 'You are good at database technologies, data types and corresponding sizing. \
    Given a list of data items that is to be data in a column of a table in a database server, you respond correctively with \
    applicable datatype and size details the column should have to store the data items as a column in a table \
    of a database server. \
    Example of data type with size details are as follows in back ticks: \
    `Decimal(10,2)`\
    `Varchar(MAX)`\
    `NVARCHAR(30)`\
    You respond in JSON format with 2 key-value pairs: Column name, suggested Data type . \
    If list has no data or lots of Nan for items, respond with "Cannot determine data type" as suggested Data type ' 
    )

    template_for_system_message_prompt = SystemMessagePromptTemplate(prompt = sys_prompt_temp)

    # create human template
    hum_prompt_temp =  PromptTemplate.from_template(
    template = 'Can you please determine the data type to use to store this list of items - {input_data_list} as a column \
    in a database table in a {database_technology} server?'
    )

    template_for_human_prompt = HumanMessagePromptTemplate(prompt = hum_prompt_temp)
    
    # chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages([
    template_for_system_message_prompt,
    template_for_human_prompt
    ])

    # create the LLM chain
    llm_chain = LLMChain(llm=chat_model, prompt=chat_prompt)

    # columns of the input dataframe as a list
    col_list = df.columns.tolist()

    # empty list to store the dictionary responses of column names and suggested data types
    list_col_dt = []

    # counters for token counts and cost
    total_tokens, prompt_tokens, completion_tokens, total_cost = 0,0,0,0

    # write token data
    placeholder = st.empty()               

    for col in col_list:
        # message to token metrics
        placeholder.text('Total Tokens: {0}\n \
                   Prompt Tokens: {1}\n \
                   Completion Tokens: {2}\n \
                   Total Cost: ${3:.2f}'.format(total_tokens, prompt_tokens, completion_tokens, total_cost))


        # pass content to llm and measure tokens/cost
        with get_openai_callback() as cb:
            response = llm_chain.run(database_technology=dbt, input_data_list = df[col])
            datatype_dict = eval(response)
            list_col_dt.append(datatype_dict)
            total_tokens += cb.total_tokens
            prompt_tokens += cb.prompt_tokens
            completion_tokens += cb.completion_tokens
            total_cost += cb.total_cost

    
    # create a dataframe based on list of responses from LLM
    output_df = pd.DataFrame.from_dict(list_col_dt)

    return output_df, total_tokens, prompt_tokens, completion_tokens, total_cost













    



if __name__ == '__main__': main()
