import LeagueTabs from "@/components/LeagueTabs";
import World from "@/components/World";
import { getGlobal, availableLeagues } from "@/lib/data";

export const metadata = { title: "Comparación entre ligas" };
export default function Page() { return (<><LeagueTabs current="mundo" leagues={availableLeagues()} /><World G={getGlobal()} /></>); }
