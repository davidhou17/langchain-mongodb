import { formatDocumentsAsString } from "langchain/util/document";
import { MongoClient } from "mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { PromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import * as fs from 'fs';

process.env.OPENAI_API_KEY = "<API-key>";
process.env.ATLAS_CONNECTION_STRING = "<connection-string>";
const client = new MongoClient(process.env.ATLAS_CONNECTION_STRING);

// Create the vector store
async function run() {
 try {
   // Save online PDF to file
   const rawData = await fetch("<online-pdf-link>");
   const pdfBuffer = await rawData.arrayBuffer();
   const pdfData = Buffer.from(pdfBuffer);
   fs.writeFileSync("test.pdf", pdfData);

   // Load and split the sample data
   const loader = new PDFLoader(`test.pdf`);
   const data = await loader.load();
   const textSplitter = new RecursiveCharacterTextSplitter({
     chunkSize: 200,
     chunkOverlap: 20,
   });

   const docs = await textSplitter.splitDocuments(data);

   // Connect to your Atlas cluster
   const database = client.db("langchain_db");
   const collection = database.collection("test");
   const dbConfig = {  
    collection: collection,
    indexName: "vector_index", // The name of the Atlas search index to use.
    textKey: "text", // The name of the collection field containing the raw content. Defaults to "text".
    embeddingKey: "embedding", // The name of the collection field containing the embedded text. Defaults to "embedding".
   };
   
   // Instantiate Atlas as the vector store
   const vectorStore = await MongoDBAtlasVectorSearch.fromDocuments(docs, new OpenAIEmbeddings(), dbConfig);

   // Define your Atlas Search Index
   const vectorSearchIndex = {
    name: "vector_index",
    type: "vectorSearch",
    definition: {
       "fields":[
          {
             "type": "vector",
             "path": "embedding",
             "numDimensions": 1536,
             "similarity": "cosine"
          }
        ]
      }
   };
   await collection.createSearchIndex(vectorSearchIndex);
   console.log("Starting index build...");

   // Wait for Atlas to build the index
   let getIndex = await collection.listSearchIndexes("vector_index").toArray();
   let indexStatus = getIndex[0].status;

   while (indexStatus != "READY") {
    console.log("...");
    getIndex = await collection.listSearchIndexes("vector_index").toArray();
    indexStatus = getIndex[0].status;
   }
   console.log(indexStatus);

   // Run vector search queries on your data
   const query = "<query>";

   // Basic semantic search query
   const basicResults = await vectorStore.similaritySearch(query, 3);
   console.log("Semantic Search Results:")
   console.log(JSON.stringify(basicResults))

   // Max Marginal Relevance search query (with metadata pre-filtering)

   const mmrResults = await vectorStore.maxMarginalRelevanceSearch(query, {
    k: 3, // Return the top 3 documents
    fetchK: 10, // The number of documents to return on initial fetch
   });
   console.log("Marginal Relevance Search Results:")
   console.log(JSON.stringify(mmrResults))

   // Implement RAG to answer questions on your data 
   const retriever = vectorStore.asRetriever();
   const model = new ChatOpenAI({});
   const prompt =
     PromptTemplate.fromTemplate(`Answer the question based only on the following context:
     {context}

     Question: {question}`);

   const chain = RunnableSequence.from([
     {
       context: retriever.pipe(formatDocumentsAsString),
       question: new RunnablePassthrough(),
     },
     prompt,
     model,
     new StringOutputParser(),
   ]);

   const question = "<question>";  
   const answer = await chain.invoke(question);
   console.log("Question: " + question);
   console.log("Answer: " + answer);

 } finally {
   // Ensures that the client will close when you finish/error
   await client.close();
}
}
run().catch(console.dir);
