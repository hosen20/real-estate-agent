# real-estate-agent
### Project description
Finding a house with a good price can be hard, the time also spent on studying the price trends according to the characteristics of the house takes time and effort. This tool helps decrease the time and effort required for understanding and finding the price of a house according to given information and characteristics about it. This agent not only provides the price but also insights about what drives this price.
### Setup instructions
1. Use the folder named "render_railway_deployment" to deploy the online dockerized API.
2. Add an API key for Groq LLM in the settings since the API calls Groq.
3. For railway use $PORT instead of a fixed port number in the DockerFile.
4. Deploy streamlit online by connecting streamlit to the file in the streamlit folder.
5. In settings add the url for the deployed API for streamlit to be able to fetch results. 
### Environment variables
1. DEPLOYMENT_URL
2. GROQ_API_KEY
