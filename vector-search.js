#!/usr/bin/env node
/**
 * 轻量级向量检索引擎 (No Dependency)
 * 
 * 功能：
 * 1. 将小说章节/设定切片为向量
 * 2. 支持语义搜索（无需外部 API，使用 TF-IDF + 余弦相似度）
 * 3. 自动维护索引文件
 * 
 * 使用场景：
 * - 写第 100 章时，搜索"主角第一次突破筑基期"的相关段落
 * - 检查伏笔时，搜索所有提到"神秘玉佩"的章节
 */

const fs = require('fs');
const path = require('path');

// ==================== 配置 ====================
const INDEX_DIR = path.join(process.env.HOME || process.env.USERPROFILE, '.novel_index');
const STOP_WORDS = new Set([
  '的', '了', '是', '在', '我', '有', '和', '就', '不', '人',
  '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
  '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他'
]);

// ==================== 工具函数 ====================

/**
 * 中文分词 (简化版：按字 + 常用词)
 */
function tokenize(text) {
  // 移除标点
  const clean = text.replace(/[^\w\u4e00-\u9fa5]/g, ' ');
  // 简单分词：2 字组合 + 单字
  const words = [];
  for (let i = 0; i < clean.length - 1; i++) {
    const twoChar = clean.slice(i, i + 2);
    if (/[\u4e00-\u9fa5]{2}/.test(twoChar)) {
      words.push(twoChar);
    }
  }
  // 过滤停用词
  return words.filter(w => !STOP_WORDS.has(w) && w.trim());
}

/**
 * 计算 TF-IDF 向量
 */
function computeTFIDF(documents, query) {
  const docCount = documents.length;
  const queryTokens = tokenize(query);
  
  // 计算词频 (TF)
  const tf = documents.map(doc => {
    const tokens = tokenize(doc);
    const freq = {};
    tokens.forEach(t => freq[t] = (freq[t] || 0) + 1);
    // 归一化
    const maxFreq = Math.max(...Object.values(freq), 1);
    Object.keys(freq).forEach(k => freq[k] /= maxFreq);
    return freq;
  });
  
  // 计算逆文档频率 (IDF)
  const idf = {};
  const allTokens = new Set([...documents.flatMap(d => tokenize(d)), ...queryTokens]);
  allTokens.forEach(token => {
    const docsWithToken = documents.filter(doc => tokenize(doc).includes(token)).length;
    idf[token] = Math.log((docCount + 1) / (docsWithToken + 1)) + 1;
  });
  
  // 计算查询向量
  const queryVec = {};
  queryTokens.forEach(t => {
    if (idf[t]) queryVec[t] = (queryVec[t] || 0) + idf[t];
  });
  
  // 计算文档向量与查询的余弦相似度
  const scores = tf.map((docFreq, idx) => {
    let dotProduct = 0;
    let queryNorm = 0;
    let docNorm = 0;
    
    Object.keys(queryVec).forEach(token => {
      const qVal = queryVec[token];
      const dVal = docFreq[token] || 0;
      dotProduct += qVal * dVal;
      queryNorm += qVal * qVal;
      docNorm += dVal * dVal;
    });
    
    queryNorm = Math.sqrt(queryNorm);
    docNorm = Math.sqrt(docNorm);
    
    if (queryNorm === 0 || docNorm === 0) return 0;
    return dotProduct / (queryNorm * docNorm);
  });
  
  return scores;
}

// ==================== 核心 API ====================

/**
 * 初始化索引目录
 */
function initIndex() {
  if (!fs.existsSync(INDEX_DIR)) {
    fs.mkdirSync(INDEX_DIR, { recursive: true });
  }
}

/**
 * 添加/更新文档到索引
 * @param {string} novelId - 小说 ID (书名)
 * @param {string} docId - 文档 ID (章节名)
 * @param {string} content - 文档内容
 */
function addDocument(novelId, docId, content) {
  initIndex();
  const indexPath = path.join(INDEX_DIR, `${novelId}.json`);
  
  let indexData = { documents: {}, metadata: {} };
  if (fs.existsSync(indexPath)) {
    indexData = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
  }
  
  indexData.documents[docId] = content;
  indexData.metadata[docId] = {
    updatedAt: Date.now(),
    wordCount: content.length
  };
  
  fs.writeFileSync(indexPath, JSON.stringify(indexData, null, 2));
  console.log(`✅ 索引更新：${novelId}/${docId}`);
}

/**
 * 搜索相关文档
 * @param {string} novelId - 小说 ID
 * @param {string} query - 搜索词
 * @param {number} topK - 返回结果数量
 * @returns {Array<{docId: string, content: string, score: number}>}
 */
function search(novelId, query, topK = 5) {
  initIndex();
  const indexPath = path.join(INDEX_DIR, `${novelId}.json`);
  
  if (!fs.existsSync(indexPath)) {
    console.warn(`⚠️ 未找到小说索引：${novelId}`);
    return [];
  }
  
  const indexData = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
  const documents = indexData.documents;
  const docIds = Object.keys(documents);
  const contents = docIds.map(id => documents[id]);
  
  const scores = computeTFIDF(contents, query);
  
  // 排序并返回 TopK
  const results = docIds
    .map((id, idx) => ({
      docId: id,
      content: documents[id].slice(0, 500), // 只返回前 500 字预览
      score: scores[idx]
    }))
    .filter(r => r.score > 0.05) // 过滤低相关度
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
  
  return results;
}

/**
 * 获取所有已索引的小说
 */
function listNovels() {
  initIndex();
  const files = fs.readdirSync(INDEX_DIR).filter(f => f.endsWith('.json'));
  return files.map(f => f.replace('.json', ''));
}

// ==================== CLI 入口 ====================
if (require.main === module) {
  const [,, action, ...args] = process.argv;
  
  switch (action) {
    case 'add':
      // node vector-search.js add novelId docId content
      const [novelId, docId, ...contentParts] = args;
      addDocument(novelId, docId, contentParts.join(' '));
      break;
    
    case 'search':
      // node vector-search.js search novelId query
      const [sNovelId, ...queryParts] = args;
      const results = search(sNovelId, queryParts.join(' '));
      console.log(JSON.stringify(results, null, 2));
      break;
    
    case 'list':
      console.log(listNovels());
      break;
    
    default:
      console.log('用法: node vector-search.js [add|search|list] ...');
  }
}

// 导出为模块
module.exports = { initIndex, addDocument, search, listNovels };
