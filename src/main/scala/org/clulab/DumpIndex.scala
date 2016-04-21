package org.clulab

import com.typesafe.config.ConfigFactory
import edu.arizona.sista.reach.nxml.indexer.NxmlSearcher
import edu.arizona.sista.embeddings.word2vec.Word2Vec
import edu.arizona.sista.reach.PaperReader
import edu.arizona.sista.processors.Document
import org.apache.lucene.document.{Document => LuceneDocument}
import scala.collection.parallel.ForkJoinTaskSupport
import java.util.zip.GZIPOutputStream
import org.apache.commons.compress.compressors.gzip._
import java.io.{FileOutputStream, Writer, OutputStreamWriter, File}


/**
 * Takes a path to a lucene index, <br>
 * tokenizes the text of each doc using BioNLPProcessor, <br>
 * sanitizes each word (prep for for w2v), <br>
 * and writes each doc to a .gz file where each line is a tokenized sentence of sanitized tokens
 */
object DumpIndex extends App {

  val config = ConfigFactory.load()
  val indexDir = config.getString("indexDir")
  val searcher = new NxmlSearcher(indexDir)
  val threadLimit = config.getInt("threadLimit")
  val bioproc = PaperReader.rs.processor
  val outDir = config.getString("indexDump")

  /** Dumps processed text to paperid.txt.gz file */
  def writeToCompressedFile(text: String, outFile: String): Unit = {
    try {
      val output: FileOutputStream = new FileOutputStream(outFile)
      val writer: Writer = new OutputStreamWriter(new GZIPOutputStream(output), "UTF-8")
      writer.write(text)
      writer.close()
      output.close()
    } catch {
      case e: Exception => println(s"Couldn't write $outFile")
    }
  }

  /** Prepares text of LuceneDocument for input to embedding generation procedure <br>
    * Dumps text to paperid.txt.gz file
    * */
  def processEntry(entry: LuceneDocument): Unit = {
    // get text and id
    val text = entry.getField("text").stringValue
    val pmid = entry.getField("id").stringValue
    println(s"Processing $pmid ...")
    // tokenize
    val doc = tokenize(text)
    val outFile = new File(outDir, s"$pmid.txt")
    // iterate over each sentence
    val sanitizedLines: Seq[String] = doc.sentences.map{ s =>
      // sanitize each word
      s.words.map(w => Word2Vec.sanitizeWord(w)).mkString(" ")
    }
    // write to disk...
    val gzipOutFile = GzipUtils.getCompressedFilename(outFile.getAbsolutePath)
    println(s"writing $gzipOutFile ...")
    writeToCompressedFile(sanitizedLines.mkString("\n"), gzipOutFile)
  }

  /** Split sentences and tokenize */
  def tokenize(text: String): Document = bioproc.mkDocument(text)

  /** Process each doc in Lucene index */
  def dumpFilesFromIndex(searcher: NxmlSearcher, nThreads: Int): Unit = {
    val docs = (0 until searcher.reader.maxDoc).par
    // limit threads
    docs.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(nThreads))
    for {
      // limited parallel iteration over the indexed documents
      i <- docs
    } {
      val entry = searcher.reader.document(i)
      processEntry(entry)
    }
  }
  println("Processing lucene documents ...")
  // prepare indexed papers for generation of embeddings
  dumpFilesFromIndex(searcher, threadLimit)
}