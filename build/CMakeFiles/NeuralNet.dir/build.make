# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/default/Desktop/neat

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/default/Desktop/neat/build

# Include any dependencies generated for this target.
include CMakeFiles/NeuralNet.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/NeuralNet.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/NeuralNet.dir/flags.make

CMakeFiles/NeuralNet.dir/main.cpp.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/default/Desktop/neat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/NeuralNet.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NeuralNet.dir/main.cpp.o -c /home/default/Desktop/neat/main.cpp

CMakeFiles/NeuralNet.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNet.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/default/Desktop/neat/main.cpp > CMakeFiles/NeuralNet.dir/main.cpp.i

CMakeFiles/NeuralNet.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNet.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/default/Desktop/neat/main.cpp -o CMakeFiles/NeuralNet.dir/main.cpp.s

CMakeFiles/NeuralNet.dir/src/layer.cpp.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/src/layer.cpp.o: ../src/layer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/default/Desktop/neat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/NeuralNet.dir/src/layer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NeuralNet.dir/src/layer.cpp.o -c /home/default/Desktop/neat/src/layer.cpp

CMakeFiles/NeuralNet.dir/src/layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNet.dir/src/layer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/default/Desktop/neat/src/layer.cpp > CMakeFiles/NeuralNet.dir/src/layer.cpp.i

CMakeFiles/NeuralNet.dir/src/layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNet.dir/src/layer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/default/Desktop/neat/src/layer.cpp -o CMakeFiles/NeuralNet.dir/src/layer.cpp.s

CMakeFiles/NeuralNet.dir/src/neural_network.cpp.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/src/neural_network.cpp.o: ../src/neural_network.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/default/Desktop/neat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/NeuralNet.dir/src/neural_network.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NeuralNet.dir/src/neural_network.cpp.o -c /home/default/Desktop/neat/src/neural_network.cpp

CMakeFiles/NeuralNet.dir/src/neural_network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNet.dir/src/neural_network.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/default/Desktop/neat/src/neural_network.cpp > CMakeFiles/NeuralNet.dir/src/neural_network.cpp.i

CMakeFiles/NeuralNet.dir/src/neural_network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNet.dir/src/neural_network.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/default/Desktop/neat/src/neural_network.cpp -o CMakeFiles/NeuralNet.dir/src/neural_network.cpp.s

CMakeFiles/NeuralNet.dir/src/activation.cpp.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/src/activation.cpp.o: ../src/activation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/default/Desktop/neat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/NeuralNet.dir/src/activation.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NeuralNet.dir/src/activation.cpp.o -c /home/default/Desktop/neat/src/activation.cpp

CMakeFiles/NeuralNet.dir/src/activation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNet.dir/src/activation.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/default/Desktop/neat/src/activation.cpp > CMakeFiles/NeuralNet.dir/src/activation.cpp.i

CMakeFiles/NeuralNet.dir/src/activation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNet.dir/src/activation.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/default/Desktop/neat/src/activation.cpp -o CMakeFiles/NeuralNet.dir/src/activation.cpp.s

CMakeFiles/NeuralNet.dir/src/loss.cpp.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/src/loss.cpp.o: ../src/loss.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/default/Desktop/neat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/NeuralNet.dir/src/loss.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NeuralNet.dir/src/loss.cpp.o -c /home/default/Desktop/neat/src/loss.cpp

CMakeFiles/NeuralNet.dir/src/loss.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNet.dir/src/loss.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/default/Desktop/neat/src/loss.cpp > CMakeFiles/NeuralNet.dir/src/loss.cpp.i

CMakeFiles/NeuralNet.dir/src/loss.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNet.dir/src/loss.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/default/Desktop/neat/src/loss.cpp -o CMakeFiles/NeuralNet.dir/src/loss.cpp.s

CMakeFiles/NeuralNet.dir/src/utils.cpp.o: CMakeFiles/NeuralNet.dir/flags.make
CMakeFiles/NeuralNet.dir/src/utils.cpp.o: ../src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/default/Desktop/neat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/NeuralNet.dir/src/utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/NeuralNet.dir/src/utils.cpp.o -c /home/default/Desktop/neat/src/utils.cpp

CMakeFiles/NeuralNet.dir/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/NeuralNet.dir/src/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/default/Desktop/neat/src/utils.cpp > CMakeFiles/NeuralNet.dir/src/utils.cpp.i

CMakeFiles/NeuralNet.dir/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/NeuralNet.dir/src/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/default/Desktop/neat/src/utils.cpp -o CMakeFiles/NeuralNet.dir/src/utils.cpp.s

# Object files for target NeuralNet
NeuralNet_OBJECTS = \
"CMakeFiles/NeuralNet.dir/main.cpp.o" \
"CMakeFiles/NeuralNet.dir/src/layer.cpp.o" \
"CMakeFiles/NeuralNet.dir/src/neural_network.cpp.o" \
"CMakeFiles/NeuralNet.dir/src/activation.cpp.o" \
"CMakeFiles/NeuralNet.dir/src/loss.cpp.o" \
"CMakeFiles/NeuralNet.dir/src/utils.cpp.o"

# External object files for target NeuralNet
NeuralNet_EXTERNAL_OBJECTS =

NeuralNet: CMakeFiles/NeuralNet.dir/main.cpp.o
NeuralNet: CMakeFiles/NeuralNet.dir/src/layer.cpp.o
NeuralNet: CMakeFiles/NeuralNet.dir/src/neural_network.cpp.o
NeuralNet: CMakeFiles/NeuralNet.dir/src/activation.cpp.o
NeuralNet: CMakeFiles/NeuralNet.dir/src/loss.cpp.o
NeuralNet: CMakeFiles/NeuralNet.dir/src/utils.cpp.o
NeuralNet: CMakeFiles/NeuralNet.dir/build.make
NeuralNet: CMakeFiles/NeuralNet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/default/Desktop/neat/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable NeuralNet"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/NeuralNet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/NeuralNet.dir/build: NeuralNet

.PHONY : CMakeFiles/NeuralNet.dir/build

CMakeFiles/NeuralNet.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/NeuralNet.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NeuralNet.dir/clean

CMakeFiles/NeuralNet.dir/depend:
	cd /home/default/Desktop/neat/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/default/Desktop/neat /home/default/Desktop/neat /home/default/Desktop/neat/build /home/default/Desktop/neat/build /home/default/Desktop/neat/build/CMakeFiles/NeuralNet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/NeuralNet.dir/depend

