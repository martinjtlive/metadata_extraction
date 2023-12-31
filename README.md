# Problem Statement
Often to load a delimited file into a target table in any db server, we need to review the data of each column of the delimited file. Then we can determine what the datatype needs to be for each column and what would be the related sizing details for the data types.

# Solution approach
This project lets us input a delimited file and the database technology of interest. Using these inputs, we pass the data into AzureChatOpenAI. It ouputs the suggested datatype and sizing details for each column of data in the delimited file based on the database techonology of the target table.

The delimited file is assumed to be .csv. 

Langchain library is used as wrapper around AzureChatOpenAI.

## Improvements to do:
1. **Not done as of 2023-12-01** Input data of delimited file is sent to the LLM 1 column at a time. If the column has a lot of data, then the token limit per request to the LLM can go over. The improvement to do would be to chunk the data for a column of data to be within the token limit, then pass to the LLM. The responses for each chunk of the same column are aggregated against the same column reference. Off this, the appropriate suggested response is to be taken for the column. Perhaps the response for each chunk is placed in the conversation buffer and then the LLM can suggest the best suggested datatype and sizing off the responses in the conversation buffer for all the chunks associated with a column of data.



