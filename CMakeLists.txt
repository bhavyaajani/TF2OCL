cmake_minimum_required(VERSION 3.9)

# common options

enable_language(CXX)

# project configuration

project(tf2ocl)

set(USE_MKL false CACHE BOOL "Flag to use Intel's MKL")

if(USE_MKL)
  if(DEFINED MKL_DIR)
    add_definitions(-DUSE_MKL)
    set(MKL_INCLUDE_DIR "${MKL_DIR}/include" CACHE PATH "" FORCE)
	#file(GLOB_RECURSE MKL_LIBS "${MKL_DIR}/lib/intel64/*.lib")
	set(MKL_LIBS "${MKL_DIR}/lib/intel64/mkl_core.lib" "${MKL_DIR}/lib/intel64/mkl_rt.lib")
  else()
	set(MKL_DIR "" CACHE PATH "")
	message(FATAL_ERROR "MKL_DIR not found")
  endif()
else()
  set(MKL_INCLUDE_DIR "" CACHE PATH "")
  set(MKL_LIBS "" CACHE PATH "")
endif()

find_path(OpenCL_INCLUDE_DIR 
	NAMES CL/cl.h NO_DEFAULT_PATH
	PATHS ENV "PROGRAMFILES(X86)"
	PATH_SUFFIXES "IntelSWTools/OpenCL/sdk/include/" )

 find_library(OpenCL_LIBRARY
      NAMES OpenCL
      NO_DEFAULT_PATH 
      PATHS
        ENV "PROGRAMFILES(X86)"
      PATH_SUFFIXES
        "IntelSWTools/OpenCL/sdk/lib/x64/")
		
include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)

include_directories("./third_party")
include_directories(tf2ocl PRIVATE "./include" ${OpenCL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${MKL_INCLUDE_DIR})

file(GLOB_RECURSE HXX_SOURCES "include/*.h")
file(GLOB_RECURSE CXX_SOURCES "src/*.cxx")
file(GLOB_RECURSE CL_SOURCES "src/*.cl")

add_executable(tf2ocl "")
target_sources(tf2ocl PRIVATE ${CXX_SOURCES} ${CL_SOURCES} ${HXX_SOURCES})
target_link_libraries(tf2ocl ${OpenCL_LIBRARY} ${OpenCV_LIBS} ${MKL_LIBS})

add_custom_command(TARGET tf2ocl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CL_SOURCES} ${PROJECT_BINARY_DIR}
)