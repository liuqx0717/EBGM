if [ ! -d "build-debug" ]; then
  mkdir build-debug
fi

cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug ../src

