𝗔𝗻 𝗘𝗻𝗱-𝘁𝗼-𝗘𝗻𝗱 𝗥𝗔𝗚 based 𝗖𝗵𝗮𝘁𝗯𝗼𝘁, built with 𝗔𝗪𝗦 𝗕𝗲𝗱𝗿𝗼𝗰𝗸 & 𝗟𝗮𝗻𝗴𝗖𝗵𝗮𝗶𝗻! 🙌



After a lot of learning and hands-on experimentation, I’ve built and deployed a fully functional 𝗥𝗲𝘁𝗿𝗶𝗲𝘃𝗮𝗹-𝗔𝘂𝗴𝗺𝗲𝗻𝘁𝗲𝗱 𝗚𝗲𝗻𝗲𝗿𝗮𝘁𝗶𝗼𝗻 (𝗥𝗔𝗚) chatbot. It combines some of the coolest tech out there—𝗟𝗮𝗻𝗴𝗖𝗵𝗮𝗶𝗻 for document processing and AWS Bedrock for AI-powered embeddings and text generation, all wrapped up in a clean and simple Streamlit interface.



 𝗪𝗵𝗮𝘁 𝗗𝗼𝗲𝘀 𝘁𝗵𝗲 𝗔𝗽𝗽 𝗗𝗼?

- 𝗣𝗗𝗙 𝗨𝗽𝗹𝗼𝗮𝗱 & 𝗣𝗿𝗼𝗰𝗲𝘀𝘀𝗶𝗻𝗴: You can upload any PDFs, and the app will break them down into manageable chunks for easy information retrieval. 

- 𝗔𝘂𝘁𝗼𝗺𝗮𝘁𝗶𝗰 𝗩𝗲𝗰𝘁𝗼𝗿 𝗦𝘁𝗼𝗿𝗲 𝗖𝗿𝗲𝗮𝘁𝗶𝗼𝗻: Once the PDFs are uploaded, the app automatically processes the text and creates a vector store using FAISS. No manual steps needed!

- 𝗔𝘀𝗸 𝗤𝘂𝗲𝘀𝘁𝗶𝗼𝗻𝘀 𝗶𝗻 𝗥𝗲𝗮𝗹 𝗧𝗶𝗺𝗲: You can then ask questions based on the content of the PDFs. The app fetches the most relevant chunks and generates detailed answers using AWS Bedrock’s Titan models.

- 𝗦𝘁𝗿𝗲𝗮𝗺𝗹𝗶𝘁 𝗨𝗜: It’s all packaged in an easy-to-use Streamlit app with real-time feedback on actions like uploading files and generating answers.



 The Tech Behind the Scenes:

- 𝗟𝗟𝗠 𝗨𝘀𝗲𝗱: The app leverages 𝗔𝗺𝗮𝘇𝗼𝗻 𝗧𝗶𝘁𝗮𝗻 𝗧𝗲𝘅𝘁 𝗘𝘅𝗽𝗿𝗲𝘀𝘀 𝘃𝟭 𝗳𝗼𝗿 𝘁𝗲𝘅𝘁 𝗴𝗲𝗻𝗲𝗿𝗮𝘁𝗶𝗼𝗻, which is a robust language model offered by AWS Bedrock. It can handle context-aware question answering and ensures detailed responses with a max token limit of 512, adjustable temperature, and topP sampling for high-quality outputs.

- 𝗔𝗪𝗦 𝗕𝗲𝗱𝗿𝗼𝗰𝗸 provides the Amazon Titan models for text embedding and generation, ensuring high-quality document processing and response generation.

- 𝗟𝗮𝗻𝗴𝗖𝗵𝗮𝗶𝗻 does the heavy lifting by splitting the documents, embedding them, and connecting with FAISS for similarity search.

- 𝗙𝗔𝗜𝗦𝗦 𝗩𝗲𝗰𝘁𝗼𝗿 𝗦𝘁𝗼𝗿𝗲: The app uses this to store document embeddings for fast, accurate retrieval.

- Streamlit Deployment: The user interface is simple and intuitive, allowing you to upload PDFs and ask questions without any hassle.



💡 𝗞𝗲𝘆 𝗛𝗶𝗴𝗵𝗹𝗶𝗴𝗵𝘁𝘀:

- 𝗔𝘂𝘁𝗼𝗺𝗮𝘁𝗶𝗰𝗮𝗹𝗹𝘆 𝗽𝗿𝗼𝗰𝗲𝘀𝘀𝗲𝘀 𝗮𝗻𝗱 𝘀𝘁𝗼𝗿𝗲𝘀 𝗣𝗗𝗙𝘀 as vectors for fast retrieval.

- 𝗨𝘀𝗲𝘀 𝗥𝗔𝗚 𝘁𝗼 𝗴𝗲𝗻𝗲𝗿𝗮𝘁𝗲 𝗮𝗰𝗰𝘂𝗿𝗮𝘁𝗲, 𝗰𝗼𝗻𝘁𝗲𝘅𝘁-𝗮𝘄𝗮𝗿𝗲 𝗮𝗻𝘀𝘄𝗲𝗿𝘀 𝗯𝗮𝘀𝗲𝗱 𝗼𝗻 𝘁𝗵𝗲 𝘂𝗽𝗹𝗼𝗮𝗱𝗲𝗱 𝗱𝗼𝗰𝘂𝗺𝗲𝗻𝘁𝘀.

- 𝗦𝗲𝗮𝗺𝗹𝗲𝘀𝘀 𝗶𝗻𝘁𝗲𝗿𝗮𝗰𝘁𝗶𝗼𝗻: Just upload a PDF, ask a question, and get a detailed answer!



🌐 𝗖𝗵𝗲𝗰𝗸 𝗼𝘂𝘁 𝘁𝗵𝗲 𝗹𝗶𝘃𝗲 𝗱𝗲𝗺𝗼 𝗵𝗲𝗿𝗲: https://lnkd.in/g6JfiRiJ



