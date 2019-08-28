#pragma once

#ifndef NDEBUG

// test some matrix operations.
void test1();

// test one Garbor kernel.
void test2();

// test the generation of 40 Gabor kernels, and
// the convolution of one kernel on one image.
void test3();

// test the calculation of jets.
void test4();

// select a point (left eye), calculate its jet (called jet0).
// Then calculate the jets of the points on the same horizontal
// line. Then calculate the similarity, similarity_with_phase,
// estimated displacement between all these jets and jet0.
void test5();

// test addXXX() and replaceXXX() of Graph and GraphBunch
void test6();

// test Points class
void test7();

// test the comparation between graph and graphbunch
void test8();

// test step1
void test9();

// test step2
void test10();

// test step3
void test11();

// test step4
void test12();

// test modifyPointsOnImage()
void test13();

// test iofiles
void test14();

// test CalcJet serialization
void test15();

// test CalcJet deserialization
void test16();

// test serialization of Graph
void test17();

// test deserialization of Graph
void test18();

// test generating cache
void test19();

// test generating graph 
void test20();

// test recognition
void test21();

#endif