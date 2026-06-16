function(merge_natvis_files natvis_files out_file)

    # Prepare the output file path
    set(output_path "${CMAKE_BINARY_DIR}/${out_file}")

    # Define the path to the Python script
    set(PYTHON_SCRIPT "${CMAKE_SOURCE_DIR}/tools/merge_natvis.py")

    # Call the Python script with the list of natvis files and the output path
    add_custom_command(
        OUTPUT ${output_path}
        COMMAND ${CMAKE_COMMAND} -E echo "Concatenating .natvis files..."
        COMMAND ${CMAKE_COMMAND} -E env ${Python_EXECUTABLE} ${PYTHON_SCRIPT} -o ${output_path} ${natvis_files}
        DEPENDS ${natvis_files} ${PYTHON_SCRIPT}
    )

    # Ensure the output file is built
    add_custom_target(run_concat_natvis ALL DEPENDS ${output_path})
endfunction()
