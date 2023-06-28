import { OpenAI } from "langchain/llms";
import { RetrievalQAChain } from "langchain/chains";
import {HNSWLib} from 'langchain/vectorstores'
import {OpenAIEmbeddings} from 'langchain/embeddings'
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";   
import * as fs from 'fs'
import * as dotenv from 'dotenv'

dotenv.config()

const txtFileName = 'Candidate_list'
const question = 'Find me a candidate with three years of experience in nodejs, React, whose worked as a fullstack engineer'
const txtFilePath = `./${txtFileName}.txt`
const VECTOR_STORE_PATH = `${txtFileName}.index`
export const runWithEmbeddings = async () => {
    const model = new OpenAI({})
    let vectorstore;
    if(fs.existsSync(VECTOR_STORE_PATH)){
        console.log('Vector Exists')
        vectorstore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
    } else {
        const text = fs.readFileSync(txtFilePath, 'utf-8')
        const textSplitter = new RecursiveCharacterTextSplitter({chunkSize: 1000})
        const docs = await textSplitter.createDocuments([text])
        vectorstore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
        await vectorstore.save(VECTOR_STORE_PATH)
    }
    const chain = RetrievalQAChain.fromLLM(model, vectorstore.asRetriever());
    const res = await chain.call({
        query: question
    })
    console.log({res})
}
runWithEmbeddings()