# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_SOURCE_DIR = /home/dwu/Documents/SDC/Term2/Project4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dwu/Documents/SDC/Term2/Project4/src

# Include any dependencies generated for this target.
include CMakeFiles/pid.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pid.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pid.dir/flags.make

CMakeFiles/pid.dir/PID.cpp.o: CMakeFiles/pid.dir/flags.make
CMakeFiles/pid.dir/PID.cpp.o: PID.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dwu/Documents/SDC/Term2/Project4/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pid.dir/PID.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pid.dir/PID.cpp.o -c /home/dwu/Documents/SDC/Term2/Project4/src/PID.cpp

CMakeFiles/pid.dir/PID.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pid.dir/PID.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dwu/Documents/SDC/Term2/Project4/src/PID.cpp > CMakeFiles/pid.dir/PID.cpp.i

CMakeFiles/pid.dir/PID.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pid.dir/PID.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dwu/Documents/SDC/Term2/Project4/src/PID.cpp -o CMakeFiles/pid.dir/PID.cpp.s

CMakeFiles/pid.dir/PID.cpp.o.requires:

.PHONY : CMakeFiles/pid.dir/PID.cpp.o.requires

CMakeFiles/pid.dir/PID.cpp.o.provides: CMakeFiles/pid.dir/PID.cpp.o.requires
	$(MAKE) -f CMakeFiles/pid.dir/build.make CMakeFiles/pid.dir/PID.cpp.o.provides.build
.PHONY : CMakeFiles/pid.dir/PID.cpp.o.provides

CMakeFiles/pid.dir/PID.cpp.o.provides.build: CMakeFiles/pid.dir/PID.cpp.o


CMakeFiles/pid.dir/main.cpp.o: CMakeFiles/pid.dir/flags.make
CMakeFiles/pid.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dwu/Documents/SDC/Term2/Project4/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/pid.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pid.dir/main.cpp.o -c /home/dwu/Documents/SDC/Term2/Project4/src/main.cpp

CMakeFiles/pid.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pid.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dwu/Documents/SDC/Term2/Project4/src/main.cpp > CMakeFiles/pid.dir/main.cpp.i

CMakeFiles/pid.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pid.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dwu/Documents/SDC/Term2/Project4/src/main.cpp -o CMakeFiles/pid.dir/main.cpp.s

CMakeFiles/pid.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/pid.dir/main.cpp.o.requires

CMakeFiles/pid.dir/main.cpp.o.provides: CMakeFiles/pid.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/pid.dir/build.make CMakeFiles/pid.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/pid.dir/main.cpp.o.provides

CMakeFiles/pid.dir/main.cpp.o.provides.build: CMakeFiles/pid.dir/main.cpp.o


# Object files for target pid
pid_OBJECTS = \
"CMakeFiles/pid.dir/PID.cpp.o" \
"CMakeFiles/pid.dir/main.cpp.o"

# External object files for target pid
pid_EXTERNAL_OBJECTS =

pid: CMakeFiles/pid.dir/PID.cpp.o
pid: CMakeFiles/pid.dir/main.cpp.o
pid: CMakeFiles/pid.dir/build.make
pid: CMakeFiles/pid.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dwu/Documents/SDC/Term2/Project4/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable pid"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pid.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pid.dir/build: pid

.PHONY : CMakeFiles/pid.dir/build

CMakeFiles/pid.dir/requires: CMakeFiles/pid.dir/PID.cpp.o.requires
CMakeFiles/pid.dir/requires: CMakeFiles/pid.dir/main.cpp.o.requires

.PHONY : CMakeFiles/pid.dir/requires

CMakeFiles/pid.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pid.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pid.dir/clean

CMakeFiles/pid.dir/depend:
	cd /home/dwu/Documents/SDC/Term2/Project4/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dwu/Documents/SDC/Term2/Project4 /home/dwu/Documents/SDC/Term2/Project4 /home/dwu/Documents/SDC/Term2/Project4/src /home/dwu/Documents/SDC/Term2/Project4/src /home/dwu/Documents/SDC/Term2/Project4/src/CMakeFiles/pid.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pid.dir/depend

