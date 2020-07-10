#ifndef LANG_H_

#ifdef ANDROID_LOGCAT
#include <android/log.h>
#define LOGD(...)	__android_log_print(ANDROID_LOG_DEBUG, "LOG_TAG",__VA_ARGS__)
#else
#define LOGD(...)
#endif

namespace lang {

	template <typename T>
	struct DataStream
	{
		T* ptr;
		int len;

		T operator[] (int i) const {
			return ptr[i];
		}
	};
	using FloatStream = DataStream<float>;

	template <typename T>
	class Placement
	{
	public:
		template<typename... Args>
		void Initialize(Args... args) {
			_ptr = new(_buf) T(args...);
		}

		T* operator-> () {
			return _ptr;
		}

		const T* operator-> () const {
			return _ptr;
		}

		T& operator ()() {
			return *_ptr;
		}

		const T& operator ()() const {
			return *_ptr;
		}

		~Placement()
		{
			_ptr->~T();
		}

	private:
		int _buf[(sizeof(T) + 4) / 4];
		T *_ptr;
	};

}

#endif // LANG_H_
