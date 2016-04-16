package ucl.irdm.brexit.tcf

import java.io.{IOException, FileWriter, File}

import com.fasterxml.jackson.core.util.MinimalPrettyPrinter
import com.fasterxml.jackson.core.{TreeNode, JsonToken, JsonFactory}
import com.fasterxml.jackson.databind.{MappingJsonFactory, JsonNode, ObjectMapper}
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import twitter4j.Status

/**
  * Created by Gabriel on 4/13/2016.
  */
object JsonToOneLine {

  def processArgs(args: Array[String]): Option[(File, File)] = {
    try {
      val fin = new File(args(0))
      val fout = new File(args(1))

      if (!fin.exists())
        None
      else {
        if (!fout.exists()) fout.createNewFile()
        Some(fin, fout)
      }
    }
    catch {
      case _: Throwable => None
    }
  }

  val usage: String = "JsonToOneLine <input_file> <output_file>"

  def jsonToOneLine(fin: File, fout: File): Unit = {
    val factory = new MappingJsonFactory()
    val inParser = factory.createParser(fin)
    val outGen = factory.createGenerator(new FileWriter(fout))
    outGen.setPrettyPrinter(new MinimalPrettyPrinter(""))

    if (inParser.nextToken() != JsonToken.START_ARRAY) {
      println("Error: root should be array; quitting")
      return
    }

    outGen.writeStartArray()
    while (inParser.nextToken() != JsonToken.END_ARRAY) {
      val node = inParser.readValueAsTree();
      outGen writeRaw '\n'
      outGen writeObject node
    }
    outGen writeRaw '\n'
    outGen.writeEndArray()
    /*Stream.from(0)
      .map(_ => {
        val node: TreeNode = try {
          //val t = inParser.nextToken() // skip commas and other tokens between array elements
          inParser.readValueAsTree()
        }
        catch {
          case ex: IOException => null
        }
        node
      })
        .take(100)//.takeWhile(_ != null)
      .foreach(obj => {
        outGen writeRaw '\n'
        outGen writeObject obj
      })*/

    outGen.writeEndArray()

    inParser.close()
    outGen.close()
  }

  def main(args: Array[String]): Unit = {
    processArgs(args) match {
      case None => println(usage)
      case Some(params) =>
        val (fin, fout) = params
        jsonToOneLine(fin, fout)
    }
  }
}
