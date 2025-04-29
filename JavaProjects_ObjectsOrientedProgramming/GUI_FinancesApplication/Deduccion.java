public class Deduccion  {

	public double cuotaFija,
				   porcentaje,
				   liminf,
				   pagoEx;
	private Persona persona;
	
	public Deduccion(Persona persona){
		this.persona=persona;
		this.cuotaFija=0.00;
		this.porcentaje=0.0192;
	}
	public double dPermitida(){
		return (persona.getTotatlIngresos()*.10)+persona.subRetiro;
	}
	public double deduccion(){
		if(persona.getTotalDeducciones()>=this.dPermitida()){
			return this.dPermitida();
		}else{
			return persona.getTotalDeducciones();
		}
	}
	public double montoISR(){
		return persona.getTIngresosGravan()-this.deduccion();
	}
	public void bases(){
		if(this.montoISR()>=0.01&&this.montoISR()<=5952.84){
			this.cuotaFija=0.00;
			this.porcentaje=0.0192;
			this.liminf=0.01;
		}else if(this.montoISR()>=5952.85&&this.montoISR()<=50524.92){
			this.cuotaFija=114.29;
			this.porcentaje=0.0640;
			this.liminf=5952.85;
		}else if(this.montoISR()>=50524.93&&this.montoISR()<=88793.04){
			this.cuotaFija=2966.91;
			this.porcentaje=0.1088;
			this.liminf=50524.93;
		}else if(this.montoISR()>=88793.05&&this.montoISR()<=103218.00){
			this.cuotaFija=7130.48;
			this.porcentaje=0.1600;
			this.liminf=88793.05;
		}else if(this.montoISR()>=103218.01&&this.montoISR()<=123580.20){
			this.cuotaFija=9438.47;
			this.porcentaje=0.1792;
			this.liminf=103218.01;
		}else if(this.montoISR()>=123580.21&&this.montoISR()<=249243.48){
			this.cuotaFija=13087.37;
			this.porcentaje=0.2136;
			this.liminf=123580.21;
		}else if(this.montoISR()>=249243.49&&this.montoISR()<=392841.96){
			this.cuotaFija=39929.05;
			this.porcentaje=0.2352;
			this.liminf=249243.49;
		}else if(this.montoISR()>=392841.97&&this.montoISR()<=750000.00){
			this.cuotaFija=73703.41;
			this.porcentaje=0.3000;
			this.liminf=392841.97;
		}else if(this.montoISR()>=750000.01&&this.montoISR()<=1000000.00){
			this.cuotaFija=180850.82;
			this.porcentaje=0.3200;
			this.liminf=750000.01;
		}else if(this.montoISR()>=1000000.01&&this.montoISR()<=3000000.00){
			this.cuotaFija=260850.81;
			this.porcentaje=0.3400;
			this.liminf=1000000.01;
		}else if(this.montoISR()>=3000000.01){
			this.cuotaFija=940850.81;
			this.porcentaje=0.3500;
			this.liminf=3000000.01;
		}
	}
	
	public double totalPagar(){
		this.bases();
		this.pagoEx=(this.montoISR()-this.liminf)*this.porcentaje;
		return this.cuotaFija+this.pagoEx;
	}
	
}
