# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-vocab.cpp
            llama-grammar.cpp
            llama-sampling.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            ../common/json.hpp
            ../common/json-schema-to-grammar.h
            )

target_include_directories(llama PUBLIC . ../include ../common)

target_compile_features   (llama PUBLIC cxx_std_11) # don't bump

target_link_libraries(llama PUBLIC ggml)

# Integración de SentencePiece

# Incluye el directorio donde tienes sentencepiece_processor.h
# Cambia la ruta según la ubicación de tus archivos de sentencepiece
target_include_directories(llama PUBLIC "E:/dev/sentencepiece/src" "E:/dev/sentencepiece/build")


# # Añadir los archivos fuente de SentencePiece
# # Asume que los archivos .cc están en ./src
# add_library(sentencepiece
#             "E:/dev/sentencepiece/src/sentencepiece_processor.cc"
#             "E:/dev/sentencepiece/src/sentencepiece_trainer.cc"
#             "E:/dev/sentencepiece/src/bpe_model.cc"
#             "E:/dev/sentencepiece/src/builder.cc"
#             "E:/dev/sentencepiece/src/model_factory.cc"
#             "E:/dev/sentencepiece/src/model_interface.cc"
#             "E:/dev/sentencepiece/src/normalizer.cc"
#             "E:/dev/sentencepiece/src/unigram_model.cc"
#             "E:/dev/sentencepiece/src/util.cc"
#             "E:/dev/sentencepiece/src/word_model.cc"
#             )

# Enlaza SentencePiece con tu biblioteca llama
target_link_libraries(llama PUBLIC "E:/dev/sentencepiece/build/src/Release/sentencepiece.lib"
                                "E:/dev/sentencepiece/build/src/Release/sentencepiece_train.lib")

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_SHARED LLAMA_BUILD)
endif()