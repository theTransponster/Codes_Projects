public class Clock  {
	private int hr,
    			min,
    			sec;

	private String cd;

	public Clock(){
		this(10,0,0,"Greenwich"); //no se puede mandar llamar a otro constructor si no esta en la primera linea

	}

	public Clock(int hr, int min, int sec, String cd){
		this.setTime(hr,min,sec);
		this.cd=cd;
	}
	public void setTime(int hr, int min, int sec){
		if(hr>=0 && hr<24){
			this.hr=hr;
		}
		else{
			this.hr=10;
		}

		if(min>=0 && min<60){
			this.min=min;
		}
		else{
			this.min=0;
		}

		if(sec>=0 && sec<60){
			this.sec=sec;
		}
		else{
			this.sec=0;
		}
	}

	public void setCiudad(String cd){
		this.cd=cd;
	}

	public int getHours(){
		return hr;
	}
	public int getMinutes(){
		return min;
	}
	public int getSeconds(){
		return sec;
	}

	public String getCiudad(){
		return cd;
	}

	public void printTime(){
		System.out.println(this); //el toString se manda a llamar automaticamente cuando pones un println

	}

	public String toString(){
		String strTime=this.cd+" ";

		System.out.println(this.hr+":"+this.min+":"+this.sec);
		if (this.hr<10){
			strTime+="0";
		}
		strTime+=this.hr+":";		

		if (this.min<10){
			strTime+="0";
		}
		strTime+=this.min+":";		

		if (this.sec<10){
			strTime+="0";
		}
		strTime+=this.sec+":";		


		return strTime;

	}

public void incrementHours(){
this.hr=++this.hr%24;

}
	public void incrementMinutes(){
		this.min=++this.min%60;
		if(this.min==0){
			this.min=0;
			this.incrementHours();
		}

	}
	public void incrementSeconds(){
		this.sec=++this.sec%60;
		if(this.sec==60){
			this.sec=0;
			this.incrementMinutes();
		}

	}

	public boolean equals(Clock reloj){
		return this.sec==reloj.sec && this.min==reloj.min && this.hr==reloj.hr;

	}

public void makeCopy(Clock reloj){
this.hr= reloj.hr;
this.min= reloj.min;
this.sec= reloj.sec; 
this.cd= reloj.cd;
//this.setTime(reloj.hr, reloj.min, reloj.sec);
}

public Clock getCopy(){
return new Clock(this.hr, this.min, this.sec, this.cd);
}
	public int compareTo(Object a){
		Clock r2=(Clock)a;
		int secThis=this.sec+this.min*60+this.hr*3600;
		int secR2=r2.sec+r2.min*60+r2.hr*3600; 
		
		return secThis-secR2;
	}
}
 
