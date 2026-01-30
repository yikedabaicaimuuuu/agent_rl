import type { Message, Result } from '@/components/UserInput/types';

import type { ChatGPTResponse } from './type';

async function ChatGPTCall(
  input: Message[],
  currConversation: string,
  method: string,
): Promise<Result> {
  console.log('Here');
  const MODEL_NAME = 'gpt-3.5-turbo';
  const body = {
    model: MODEL_NAME,
    messages: input,
    conv_id: currConversation,
    method: method,
  };
  console.log(body);
  try {
    console.log(input);
    const res = await fetch('http://127.0.0.1:5001/get_response', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      throw new Error('Network response was not ok');
    }

    const response = (await res.json()) as ChatGPTResponse;
    if (
      !response.choices ||
      !response.choices[0] ||
      !response.choices[0].message ||
      !response.choices[0].message.content
    ) {
      throw new Error('Invalid response format');
    }

    const responseMessage = response.choices[0].message.content;

    return {
      type: 'success',
      data: responseMessage,
    };
  } catch (error) {
    console.error('There was a problem with the fetch operation:', error);
    return {
      type: 'err',
      data: 'An error occurred while fetching the response.',
    };
  }
}

export default ChatGPTCall;
