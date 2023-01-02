import { Box } from '@chakra-ui/react'
import useSWR from 'swr'

const fetcher = (url: RequestInfo | URL) => fetch(url).then(res => res.json())

export default function App() {
  const { data, error } = useSWR('http://localhost:8000/users', fetcher)

  if (error) return <div>failed to load</div>
  if (!data) return <div>loading...</div>
  // console.log(data)

  return (
    <>
      {/* <Box>{data[2].email}</Box> */}
    </>
  )
}