# from pocketsphinx import LiveSpeech
#
# speech = LiveSpeech(lm=False, keyphrase='forward', kws_threshold=1e-20)
# for phrase in speech:
#     print(phrase.segments(detailed=True))

def rotate(nums, k):
    k = k % len(nums)
    print(id(nums))
    nums[:] = nums[-k:] + nums[:-k]
    print(id(nums))
    print(nums)
    return nums


if __name__ == '__main__':
    numbers = [1, 2, 3]
    number2 = numbers[:]
    number2.append(4)
    print(numbers)
    print(number2)
    # print(id(numbers))
    # result = rotate(numbers, 1)
    # print(result)
    # print(numbers)
