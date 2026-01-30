from flask import jsonify, request
import time
import traceback
from openai import OpenAIError
import anthropic
import google.api_core.exceptions as GoogleAPIError

from src import app
from src.control.query import query, proslm_query, query_rag_service
from src.model.conversation_model import append_message_by_id


@app.route("/get_response", methods=["POST"])
def get_response_control():
    start_time = time.time()
    request_id = f"req_{int(start_time)}"

    print(f"\n[DEBUG] [{request_id}] ====== NEW REQUEST ======")

    try:
        data = request.json
        conv_id = data["conv_id"]
        method = data["method"]
        provider = data.get("provider", "openai")
        model = data.get("model")

        # Debug information
        print(f"[DEBUG] [{request_id}] Received request:")
        print(f"[DEBUG] [{request_id}] - Conversation ID: {conv_id}")
        print(f"[DEBUG] [{request_id}] - Method: {method}")
        print(f"[DEBUG] [{request_id}] - Provider: {provider}")
        print(f"[DEBUG] [{request_id}] - Model: {model}")

        # Save user message
        print(f"[DEBUG] [{request_id}] Saving user message to conversation")
        append_message_by_id(conv_id, data)

        # Extract user input
        input_text = data["messages"][-1]["content"]
        print(f"[DEBUG] [{request_id}] User input: {input_text[:50]}...")

        # Process method name
        processed_method = method
        if method == "chain-of-thought":
            processed_method = "cot"
        elif method == "rag-agent":
            # Handle RAG Agent method - uses the agentic RAG pipeline
            try:
                print(f"[DEBUG] [{request_id}] Using RAG Agent pipeline")
                result = query_rag_service(input_text, use_router=True)
                response_json = {"choices": [{"message": {"content": result}}]}

                save_res = {
                    "messages": [{"role": "bot", "content": result}],
                    "provider": "rag",
                    "model": "rag-agent",
                }

                append_message_by_id(conv_id, save_res)

                total_time = time.time() - start_time
                print(f"[DEBUG] [{request_id}] RAG Agent completed in {total_time:.2f} seconds")
                return jsonify(response_json)
            except Exception as e:
                print(f"[ERROR] [{request_id}] RAG Agent error: {str(e)}")
                return jsonify({"error": f"RAG Agent Error: {str(e)}"}), 500
        elif method == "pro-slm":
            try:
                result = proslm_query(input_text)
                response_json = {"choices": [{"message": {"content": result}}]}

                save_res = {
                    "messages": [{"role": "bot", "content": result}],
                    "provider": "openai",
                    "model": "openai",
                }

                append_message_by_id(conv_id, save_res)
                return jsonify(response_json)
            except:
                print("Error in ProSLM")

        print(f"[DEBUG] [{request_id}] Processed method: {processed_method}")

        try:
            # Get response from LLM
            print(f"[DEBUG] [{request_id}] Calling query function")
            response_message, actual_provider, actual_model = query(
                input_text, processed_method, provider, model
            )

            query_time = time.time() - start_time
            print(f"[DEBUG] [{request_id}] Query completed in {query_time:.2f} seconds")
            print(
                f"[DEBUG] [{request_id}] Response from {actual_provider} ({actual_model or 'default model'})"
            )
            print(
                f"[DEBUG] [{request_id}] Response length: {len(response_message)} characters"
            )

            # Add fallback notice if applicable
            response_content = response_message
            if "fallback" in actual_provider:
                response_content = f"[NOTE: Gemini quota exceeded, falling back to OpenAI]\n\n{response_message}"

            # Format response for frontend
            response_json = {"choices": [{"message": {"content": response_content}}]}

            # Save bot response to conversation
            print(f"[DEBUG] [{request_id}] Saving bot response to conversation")
            save_res = {
                "messages": [{"role": "bot", "content": response_content}],
                "provider": actual_provider,
                "model": actual_model,
            }
            append_message_by_id(conv_id, save_res)

            total_time = time.time() - start_time
            print(
                f"[DEBUG] [{request_id}] Total request processing time: {total_time:.2f} seconds"
            )
            print(f"[DEBUG] [{request_id}] ====== REQUEST COMPLETE ======\n")

            return jsonify(response_json)

        except OpenAIError as e:
            print(f"[ERROR] [{request_id}] OpenAI Error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"OpenAI Error: {str(e)}"}), 500
        except anthropic.APIError as e:
            # Using the correct Anthropic exception class
            print(f"[ERROR] [{request_id}] Claude Error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"Claude Error: {str(e)}"}), 500
        except GoogleAPIError.GoogleAPIError as e:
            print(f"[ERROR] [{request_id}] Gemini Error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"Gemini Error: {str(e)}"}), 500
        except Exception as e:
            print(f"[ERROR] [{request_id}] Unexpected error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        print(f"[ERROR] [{request_id}] Request processing error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
