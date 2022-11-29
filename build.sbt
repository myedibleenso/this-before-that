val root: Project = (project in file(".")).settings(
  name := "this-before-that",
  organization := "org.clulab",
  version := "0.1.0",
  scalaVersion := "2.11.8",
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation"),
  testOptions in Test += Tests.Argument("-oD")
).dependsOn(
    // use the latest reach build
    //ProjectRef(uri("git://github.com/clulab/reach.git#precedence-corpus"), "reach")
    RootProject(uri("git://github.com/clulab/reach.git#precedence-corpus"))
  )

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.2.4" % "test",
  //"org.clulab" % "reach_2.11" % "1.2.2",
  "org.clulab" % "bioresources" % "1.1.2",
  "org.clulab" %% "processors" % "5.8.2",
  "org.clulab" %% "processors" % "5.8.2" classifier "models",
  "com.typesafe" % "config" % "1.2.1",
  //"commons-io" % "commons-io" % "2.4",
  "org.apache.commons" % "commons-compress" % "1.5"
)