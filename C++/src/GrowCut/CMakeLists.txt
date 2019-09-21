cmake_minimum_required(VERSION 3.1)
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(PROJECT_NAME GrowCut)
find_package(OpenCV 4.1.0 EXACT REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR}/include)
link_directories(${OpenCV_LIBRARY_DIRS})

# プロジェクト名と使用する言語を設定
project(${PROJECT_NAME} CXX)
# a.outという実行ファイルをmain.cppとhello.cppから作成
add_executable(${PROJECT_NAME}.out ${PROJECT_NAME}.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})